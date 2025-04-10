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
import re
from typing import Any, Optional

import requests
from google.cloud import storage  # type: ignore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from sqlalchemy import RowMapping, Sequence, text


class AsyncAlloyDBVectorStore(AsyncPGVectorStore):
    """Google AlloyDB Vector Store class"""

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
        **kwargs: Any,
    ) -> list[str]:
        """Embed images and add to the table.

        Args:
            uris (list[str]): List of local image URIs to add to the table.
            metadatas (Optional[list[dict]]): List of metadatas to add to table records.
            ids: (Optional[list[str]]): List of IDs to add to table records.

        Returns:
            List of record IDs added.
        """
        encoded_images = []
        if metadatas is None:
            metadatas = [{"image_uri": uri} for uri in uris]

        for uri in uris:
            encoded_image = self._encode_image(uri)
            encoded_images.append(encoded_image)

        embeddings = self._images_embedding_helper(uris)
        ids = await self.aadd_embeddings(
            encoded_images, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        return ids

    async def __query_collection(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> Sequence[RowMapping]:
        """Perform similarity search query on database."""
        k = k if k else self.k
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function

        columns = self.metadata_columns + [
            self.id_column,
            self.content_column,
            self.embedding_column,
        ]
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        safe_filter = None
        filter_dict = None
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        param_filter = f"WHERE {safe_filter}" if safe_filter else ""
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        if not embedding and callable(inline_embed_func) and "query" in kwargs:
            query_embedding = self.embedding_service.embed_query_inline(kwargs["query"])  # type: ignore
        else:
            query_embedding = f"{[float(dimension) for dimension in embedding]}"
        stmt = f"""SELECT {column_names}, {search_function}("{self.embedding_column}", :query_embedding) as distance
        FROM "{self.schema_name}"."{self.table_name}" {param_filter} ORDER BY "{self.embedding_column}" {operator} {query_embedding} LIMIT :k;
        """
        param_dict = {"k": k}
        if filter_dict:
            param_dict.update(filter_dict)
        if self.index_query_options:
            async with self.engine.connect() as conn:
                # Set each query option individually
                for query_option in self.index_query_options.to_parameter():
                    query_options_stmt = f"SET LOCAL {query_option};"
                    await conn.execute(text(query_options_stmt))
                result = await conn.execute(text(stmt), param_dict)
                result_map = result.mappings()
                results = result_map.fetchall()
        else:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(stmt), param_dict)
                result_map = result.mappings()
                results = result_map.fetchall()
        return results

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

    async def set_maintenance_work_mem(self, num_leaves: int, vector_size: int) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        # Required index memory in MB
        buffer = 1
        index_memory_required = (
            round(50 * num_leaves * vector_size * 4 / 1024 / 1024) + buffer
        )  # Convert bytes to MB
        query = f"SET maintenance_work_mem TO '{index_memory_required} MB';"
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

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
