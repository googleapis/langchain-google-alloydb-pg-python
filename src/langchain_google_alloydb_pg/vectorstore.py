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

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_postgres import PGVectorStore

from .async_vectorstore import AsyncAlloyDBVectorStore


class AlloyDBVectorStore(PGVectorStore):
    """Google AlloyDB Vector Store class"""

    __vs: AsyncAlloyDBVectorStore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__vs = self.__PGVectorStore__vs  # type: ignore

    async def aadd_images(
        self,
        uris: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed images and add to the table."""
        return await self._engine._run_as_async(
            self.__vs.aadd_images(uris, metadatas, ids, **kwargs)
        )

    def add_images(
        self,
        uris: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed images and add to the table."""
        return self._engine._run_as_sync(
            self.__vs.aadd_images(uris, metadatas, ids, **kwargs)
        )

    def similarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on image."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_image(image_uri, k, filter, **kwargs)
        )

    async def asimilarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on image_uri."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_image(image_uri, k, filter, **kwargs)
        )

    async def aset_maintenance_work_mem(
        self, num_leaves: int, vector_size: int
    ) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        await self._engine._run_as_async(
            self.__vs.set_maintenance_work_mem(num_leaves, vector_size)
        )

    def set_maintenance_work_mem(self, num_leaves: int, vector_size: int) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        self._engine._run_as_sync(
            self.__vs.set_maintenance_work_mem(num_leaves, vector_size)
        )
