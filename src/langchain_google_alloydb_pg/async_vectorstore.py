# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

import base64
import logging
import re
from typing import Any, Optional

import numpy as np
import requests
from google.cloud import storage  # type: ignore
from langchain_core.documents import Document
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _quote_ident(ident: str) -> str:
    """Quote a PostgreSQL identifier to prevent SQL injection and syntax errors."""
    return '"' + ident.replace('"', '""') + '"'


class AsyncAlloyDBVectorStore(AsyncPGVectorStore):
    """Google AlloyDB Vector Store class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _pool_engine(self) -> Any:
        """Helper to access the underlying pool from AlloyDBEngine or PGEngine."""
        return getattr(self.engine, "_pool", self.engine)

    def _encode_image(self, uri: str) -> str:
        """Get base64 string from a image URI."""
        gcs_uri = re.match("gs://(.*?)/(.*)", uri)
        if gcs_uri:
            bucket_name, object_name = gcs_uri.groups()
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            return base64.b64encode(blob.download_as_bytes()).decode("utf-8")

        web_uri = re.match(r"^(https?://).*", uri)
        if web_uri:
            response = requests.get(uri, stream=True)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")

        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def aadd_images(
        self,
        uris: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        store_uri_only: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Embed images and add to the table.

        Args:
            uris (list[str]): List of image URIs to add to the table.
            metadatas (Optional[list[dict]]): List of metadatas to add to table records.
            ids: (Optional[list[str]]): List of IDs to add to table records.
            store_uri_only (bool): If True, stores the URI in the content column
                                   instead of the base64 encoded image. Defaults to False.
            **kwargs: Any other arguments to pass to the embedding service.

        Returns:
            List of record IDs added.
        """
        if metadatas is None:
            # Ensure URI is always in metadata if not explicitly provided elsewhere
            metadatas = [{"image_uri": uri} for uri in uris]
        elif store_uri_only:
            # If storing URI only and metadatas are provided, ensure image_uri is present
            for i, m in enumerate(metadatas):
                if "image_uri" not in m:  # Add if not already provided by user
                    m["image_uri"] = uris[i]

        texts_for_content_column: list[str]
        if store_uri_only:
            texts_for_content_column = uris
        else:
            texts_for_content_column = [self._encode_image(uri) for uri in uris]

        # Embeddings are always generated from the actual image content via URIs
        embeddings = self._images_embedding_helper(uris)

        ids = await self.aadd_embeddings(
            texts_for_content_column, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        if ids:
            return ids
        return []

    def _images_embedding_helper(self, image_uris: list[str]) -> list[list[float]]:
        # check if either `embed_images()` or `embed_image()` API is supported by the embedding service used
        if hasattr(self.embedding_service, "embed_images"):
            try:
                embeddings = self.embedding_service.embed_images(image_uris)
            except Exception as e:
                raise Exception(
                    f"Make sure your selected embedding model supports list of image URIs as input. {str(e)}"
                )
        elif hasattr(self.embedding_service, "embed_image"):
            try:
                embeddings = self.embedding_service.embed_image(image_uris)
            except Exception as e:
                raise Exception(
                    f"Make sure your selected embedding model supports list of image URIs as input. {str(e)}"
                )
        else:
            raise ValueError(
                "Please use an embedding model that supports image embedding."
            )
        return embeddings

    async def asimilarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        embedding = self._images_embedding_helper([image_uri])[0]

        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    async def aset_maintenance_work_mem(
        self, num_leaves: Optional[int], vector_size: int
    ) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        if not num_leaves:
            return
        # Required index memory in MB (minimum 10 MB for ScaNN)
        buffer = 1
        index_memory_required = max(
            10, round(50 * num_leaves * vector_size * 4 / 1024 / 1024) + buffer
        )
        async with self._pool_engine.begin() as conn:
            await conn.execute(
                text(f"SET LOCAL maintenance_work_mem TO '{index_memory_required} MB';")
            )

    set_maintenance_work_mem = aset_maintenance_work_mem

    async def aapply_vector_index(
        self,
        index: Any,
        name: Optional[str] = None,
        *,
        concurrently: bool = False,
    ) -> None:
        """Create index in the vector store table with ScaNN memory management."""
        from langchain_postgres.v2.indexes import (
            DEFAULT_INDEX_NAME_SUFFIX,
            ExactNearestNeighbor,
        )

        from .indexes import ScaNNIndex

        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return

        # Note: CREATE EXTENSION is omitted here as it requires SUPERUSER privileges.
        # Extensions should be created during database setup by an administrator.

        function = index.get_index_function()

        filter = f"WHERE ({index.partial_indexes})" if index.partial_indexes else ""
        params = "WITH " + index.index_options()
        if name is None:
            if index.name is None:
                index.name = self.table_name + DEFAULT_INDEX_NAME_SUFFIX
            name = index.name
        stmt = f'CREATE INDEX {"CONCURRENTLY" if concurrently else ""} "{name}" ON "{self.schema_name}"."{self.table_name}" USING {index.index_type} ({self.embedding_column} {function}) {params} {filter};'

        mem_query = None
        if isinstance(index, ScaNNIndex) and index.num_leaves is not None:
            num_leaves: int = index.num_leaves
            # Fetch vector_size from embedding_service if available, otherwise default to 768
            vector_size: int = 768
            if hasattr(self, "embedding_service") and hasattr(
                self.embedding_service, "embedding_size"
            ):
                vector_size = (
                    getattr(self.embedding_service, "embedding_size", 768) or 768
                )
            elif hasattr(self, "vector_size"):
                vector_size = getattr(self, "vector_size", 768) or 768
            mem_mb = max(
                10,
                round(50 * num_leaves * vector_size * 4 / 1024 / 1024) + 1,
            )
            mem_query = f"SET maintenance_work_mem TO '{mem_mb} MB';"

        if concurrently:
            async with self._pool_engine.connect() as conn:
                autocommit_conn = await conn.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                if mem_query:
                    await autocommit_conn.execute(text(mem_query))
                try:
                    await autocommit_conn.execute(text(stmt))
                finally:
                    if mem_query:
                        try:
                            await autocommit_conn.execute(
                                text("RESET maintenance_work_mem;")
                            )
                        except Exception:
                            # Preserve the original CREATE INDEX exception if the connection is broken
                            pass
        else:
            async with self._pool_engine.begin() as conn:
                if mem_query:
                    # SET LOCAL is automatically scoped to the transaction block
                    await conn.execute(
                        text(f"SET LOCAL maintenance_work_mem TO '{mem_mb} MB';")
                    )
                await conn.execute(text(stmt))

    async def ainitialize_auto_vector_embeddings(
        self,
        model_id: str,
        content_column: Optional[str] = None,
        embedding_column: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> None:
        """Asynchronously initialize auto vector embeddings.

        Args:
            model_id: The ID of the model to use for embeddings.
            content_column: Optional name of the content column. Defaults to self.content_column.
            embedding_column: Optional name of the embedding column. Defaults to self.embedding_column.
            schema_name: Optional name of the database schema. Defaults to self.schema_name.
        """
        content_col = content_column or self.content_column
        embedding_col = embedding_column or self.embedding_column
        schema = schema_name or getattr(self, "schema_name", "public")

        if not content_col:
            raise ValueError(
                "content_column must be provided or configured on the vector store."
            )
        if not embedding_col:
            raise ValueError(
                "embedding_column must be provided or configured on the vector store."
            )

        def _quote_ident(ident: str) -> str:
            return '"' + ident.replace('"', '""') + '"'

        table_identifier = (
            f"{_quote_ident(schema)}.{_quote_ident(self.table_name)}"
            if schema
            else _quote_ident(self.table_name)
        )
        query = "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
        try:
            async with self._pool_engine.connect() as conn:
                await conn.execute(
                    text(query),
                    {
                        "model_id": model_id,
                        "table_name": table_identifier,
                        "content_column": content_col,
                        "embedding_column": embedding_col,
                    },
                )
                await conn.commit()
        except Exception as e:
            if (
                "ai.initialize_embeddings" in str(e)
                or "UndefinedProcedureError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB AI extension is not installed or enabled. "
                    "Please execute 'CREATE EXTENSION IF NOT EXISTS alloydb_ai CASCADE;' on your database."
                ) from e
            raise

    async def aenable_columnar_engine(
        self,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Asynchronously add the table and its columns to the columnar engine.

        Args:
            columns: Optional list of column names to add to the columnar engine.
        """
        schema = getattr(self, "schema_name", "public") or "public"
        table_identifier = (
            f"{_quote_ident(schema)}.{_quote_ident(self.table_name)}"
            if schema
            else _quote_ident(self.table_name)
        )

        if columns:
            columns_str = ",".join(_quote_ident(c) for c in columns)
            query = "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
            params = {"table_name": table_identifier, "columns": columns_str}
        else:
            # When columns is None, we should exclude vector columns to avoid wasting columnar memory
            # Query table columns excluding the embedding_column
            query = "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
            # Fetch all columns except the vector embedding column
            async with self._pool_engine.connect() as conn:
                col_result = await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = :table_name AND table_schema = :schema AND column_name != :embed_col"
                    ),
                    {
                        "table_name": self.table_name,
                        "schema": schema,
                        "embed_col": self.embedding_column,
                    },
                )
                col_names = [row[0] for row in col_result.fetchall()]
            if col_names:
                columns_str = ",".join(_quote_ident(c) for c in col_names)
                params = {"table_name": table_identifier, "columns": columns_str}
            else:
                query = "SELECT google_columnar_engine_add(:table_name)"
                params = {"table_name": table_identifier}

        try:
            async with self._pool_engine.connect() as conn:
                await conn.execute(text(query), params)
                await conn.commit()
        except Exception as e:
            if (
                "google_columnar_engine" in str(e)
                or "UndefinedFunctionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB Columnar Engine is not installed or enabled on this instance. "
                    "Please ensure 'google_columnar_engine' is in shared_preload_libraries "
                    "and 'google_columnar_engine.enabled = on' is set in instance flags."
                ) from e
            raise

    async def aenable_auto_columnarization(self) -> None:
        """Asynchronously trigger auto-columnarization recommendations."""
        query = "SELECT google_columnar_engine_recommend('AUTO_COLUMNARIZATION')"
        try:
            async with self._pool_engine.connect() as conn:
                await conn.execute(text(query))
                await conn.commit()
        except Exception as e:
            if (
                "google_columnar_engine" in str(e)
                or "UndefinedFunctionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB Columnar Engine is not installed or enabled on this instance. "
                    "Please ensure 'google_columnar_engine' is in shared_preload_libraries "
                    "and 'google_columnar_engine.enabled = on' is set in instance flags."
                ) from e
            raise

    async def adefine_vector_assist_spec(self) -> list[dict]:
        """Asynchronously define a Vector Assist spec for the current table."""
        schema = getattr(self, "schema_name", "public") or "public"
        table_identifier = (
            f"{_quote_ident(schema)}.{_quote_ident(self.table_name)}"
            if schema
            else _quote_ident(self.table_name)
        )
        query = "SELECT * FROM vector_assist.define_spec(table_name => :table_name, vector_column_name => :embedding_column)"
        params = {
            "table_name": table_identifier,
            "embedding_column": self.embedding_column,
        }
        try:
            async with self._pool_engine.connect() as conn:
                result = await conn.execute(text(query), params)
                rows = [dict(row) for row in result.mappings()]
                await conn.commit()
                return rows
        except Exception as e:
            if (
                "vector_assist" in str(e)
                or "UndefinedSchemaError" in type(e).__name__
                or "UndefinedFunctionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB Vector Assist extension is not installed on this database. "
                    "Please execute 'CREATE EXTENSION IF NOT EXISTS vector_assist CASCADE;' as a superuser."
                ) from e
            raise

    async def aapply_vector_assist_spec(
        self, spec_id: Optional[str] = None
    ) -> list[dict]:
        """Asynchronously apply the Vector Assist spec for the current table."""
        schema = getattr(self, "schema_name", "public") or "public"
        table_identifier = (
            f"{_quote_ident(schema)}.{_quote_ident(self.table_name)}"
            if schema
            else _quote_ident(self.table_name)
        )
        if spec_id:
            query = "SELECT * FROM vector_assist.apply_spec(spec_id => :spec_id)"
            params = {"spec_id": spec_id}
        else:
            query = "SELECT * FROM vector_assist.apply_spec(table_name => :table_name, vector_column_name => :embedding_column)"
            params = {
                "table_name": table_identifier,
                "embedding_column": self.embedding_column,
            }
        try:
            async with self._pool_engine.connect() as conn:
                result = await conn.execute(text(query), params)
                rows = [dict(row) for row in result.mappings()]
                await conn.commit()
                return rows
        except Exception as e:
            if (
                "vector_assist" in str(e)
                or "UndefinedSchemaError" in type(e).__name__
                or "UndefinedFunctionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB Vector Assist extension is not installed on this database. "
                    "Please execute 'CREATE EXTENSION IF NOT EXISTS vector_assist CASCADE;' as a superuser."
                ) from e
            raise

    async def aget_vector_assist_recommendations(self) -> list[dict]:
        """Asynchronously get Vector Assist recommendations for the current table."""
        schema = getattr(self, "schema_name", "public") or "public"
        table_identifier = (
            f"{_quote_ident(schema)}.{_quote_ident(self.table_name)}"
            if schema
            else _quote_ident(self.table_name)
        )

        # Query existing spec_id from vector_assist.specs instead of defining a new spec (avoids side-effects)
        query_spec = (
            "SELECT spec_id FROM vector_assist.specs "
            "WHERE table_name = :table_name AND vector_column_name = :embedding_column "
            "ORDER BY created_at DESC LIMIT 1"
        )
        try:
            async with self._pool_engine.connect() as conn:
                spec_result = await conn.execute(
                    text(query_spec),
                    {
                        "table_name": table_identifier,
                        "embedding_column": self.embedding_column,
                    },
                )
                spec_row = spec_result.mappings().first()
                if not spec_row:
                    logger.warning(
                        "No vector assist spec found for table '%s'. "
                        "Call adefine_vector_assist_spec() first to create a spec.",
                        table_identifier,
                    )
                    return []

                spec_id = spec_row.get("spec_id")
                query = "SELECT * FROM vector_assist.get_recommendations(spec_id => :spec_id)"
                result = await conn.execute(text(query), {"spec_id": str(spec_id)})
                return [dict(row) for row in result.mappings()]
        except Exception as e:
            if (
                "vector_assist" in str(e)
                or "UndefinedSchemaError" in type(e).__name__
                or "UndefinedFunctionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "AlloyDB Vector Assist extension is not installed on this database. "
                    "Please execute 'CREATE EXTENSION IF NOT EXISTS vector_assist CASCADE;' as a superuser."
                ) from e
            raise

    def add_images(
        self,
        uris: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def similarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )
