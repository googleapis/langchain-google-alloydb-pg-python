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
from typing import List

from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from .engine import AlloyDBEngine


class AlloyDBEmbeddings(Embeddings):
    """Google AlloyDB Embeddings available via Model Endpoint Management."""

    def __init__(
        self, engine: AlloyDBEngine, model_id: str = "textembedding-gecko@001"
    ):
        """AlloyDBEmbeddings constructor.
        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        """
        self._engine = engine
        self.model_id = model_id

        # TODO: @vishwarajanand - We should validate the model_id here
        # models = AlloyDBModel(engine, model_id)
        # if not models.exists():
        #     raise IllegalArgumentError(f"Model {model_id} does not exist.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    def embed_query_inline(self, query: str) -> str:
        return f"embedding('{self.model_id}', '{query}')::vector"

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            query (str): Text to embed.

        Returns:
            List[float]: Embedding.
        """
        embeddings = await self._engine._run_as_async(self._aembed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            query (str): Text to embed.

        Returns:
            List[float]: Embedding.
        """
        return self._engine._run_as_sync(self._aembed_query(text))

    async def _aembed_query(self, query: str) -> List[float]:
        """Coroutine for generating embeddings for a given query.

        Args:
            query (str): Text to embed.

        Returns:
            List[float]: Embedding.
        """
        query = f" SELECT embedding('{self.model_id}', '{query}')::vector "
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return json.loads(results[0]["embedding"])
