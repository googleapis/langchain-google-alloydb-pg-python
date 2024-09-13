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

import json
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .engine import AlloyDBEngine
from sqlalchemy import RowMapping, text


class AlloyDBMemEmbeddings(Embeddings):
    """Google AlloyDB Model Endpoint Management for Embeddings."""

    def __init__(self, engine: AlloyDBEngine, model_id: str):
        """AlloyDBMemEmbeddings constructor.
        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        """

        self._engine = engine
        self.model_id = model_id

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    def embed_query(self, text: str) -> List[float]:
        # coro = self._getEmbeddingsFromDb(text)
        # return self._engine._run_as_sync(coro)
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    async def _getEmbeddingsFromDb(self, content: str) -> List[float]:
        query = f" SELECT embedding('{self.model_id}', '{content}')::vector "
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return json.loads(results[0]['embedding'])

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    async def aembed_query(self, text: str) -> List[float]:
        embed = await self._getEmbeddingsFromDb(text)
        return embed
        # raise NotImplementedError(
        #     "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        # )

    def embed_query_inline(self, text: str) -> str:
        return f"embedding('{self.model_id}', '{text}')::vector"
