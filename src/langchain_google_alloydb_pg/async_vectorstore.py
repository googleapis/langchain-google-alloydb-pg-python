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

import numpy as np
import requests
from google.cloud import storage  # type: ignore
from langchain_core.documents import Document
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from sqlalchemy import text


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
