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


class AsyncAlloyDBVectorStore(AsyncPGVectorStore):
    """Google AlloyDB Vector Store class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # Required index memory in MB
        buffer = 1
        index_memory_required = (
            round(50 * num_leaves * vector_size * 4 / 1024 / 1024) + buffer
        )  # Convert bytes to MB
        query = f"SET maintenance_work_mem TO '{index_memory_required} MB';"
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    set_maintenance_work_mem = aset_maintenance_work_mem

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

        table_identifier = (
            f'"{schema}"."{self.table_name}"' if schema else f'"{self.table_name}"'
        )
        query = "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
        async with self.engine.connect() as conn:
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

    async def aenable_columnar_engine(
        self,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Asynchronously add the table and its columns to the columnar engine.

        Args:
            columns: Optional list of column names to add to the columnar engine.
        """
        if columns:
            columns_str = ",".join(columns)
            query = "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
            params = {"table_name": self.table_name, "columns": columns_str}
        else:
            query = "SELECT google_columnar_engine_add(:table_name)"
            params = {"table_name": self.table_name}

        async with self.engine.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    async def aenable_auto_columnarization(self) -> None:
        """Asynchronously trigger auto-columnarization recommendations."""
        query = "SELECT google_columnar_engine_recommend('AUTO_COLUMNARIZATION')"
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adefine_vector_assist_spec(self) -> list[dict]:
        """Asynchronously define a Vector Assist spec for the current table."""
        query = "SELECT * FROM vector_assist.define_spec(table_name => :table_name, vector_column_name => :embedding_column)"
        params = {
            "table_name": self.table_name,
            "embedding_column": self.embedding_column,
        }
        async with self.engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector_assist"))
            result = await conn.execute(text(query), params)
            rows = [dict(row) for row in result.mappings()]
            await conn.commit()
            return rows

    async def aapply_vector_assist_spec(
        self, spec_id: Optional[str] = None
    ) -> list[dict]:
        """Asynchronously apply the Vector Assist spec for the current table."""
        if spec_id:
            query = "SELECT * FROM vector_assist.apply_spec(spec_id => :spec_id)"
            params = {"spec_id": spec_id}
        else:
            query = "SELECT * FROM vector_assist.apply_spec(table_name => :table_name, column_name => :embedding_column)"
            params = {
                "table_name": self.table_name,
                "embedding_column": self.embedding_column,
            }
        async with self.engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector_assist"))
            result = await conn.execute(text(query), params)
            rows = [dict(row) for row in result.mappings()]
            await conn.commit()
            return rows

    async def aget_vector_assist_recommendations(self) -> list[dict]:
        """Asynchronously get Vector Assist recommendations for the current table."""
        # First we need to get the spec ID for the current table
        specs = await self.adefine_vector_assist_spec()
        if not specs:
            logger.warning(
                "No vector assist spec found for table '%s'.", self.table_name
            )
            return []

        spec_id = specs[0].get("vector_spec_id")
        if spec_id is None:
            logger.warning(
                "Vector assist spec for table '%s' does not contain 'vector_spec_id'.",
                self.table_name,
            )
            return []

        query = "SELECT * FROM vector_assist.get_recommendations(spec_id => :spec_id)"
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), {"spec_id": str(spec_id)})
            return [dict(row) for row in result.mappings()]

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
