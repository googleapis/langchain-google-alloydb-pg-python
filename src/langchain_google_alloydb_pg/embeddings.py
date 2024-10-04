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

import json
from typing import List

from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from .engine import AlloyDBEngine
from .model_manager import AlloyDBModelManager


class AlloyDBEmbeddings(Embeddings):
    """Google AlloyDB Embeddings available via Model Endpoint Management."""

    def __init__(self, engine: AlloyDBEngine, model_id: str):
        """AlloyDBEmbeddings constructor.
        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        Raises:
            :class:`ValueError`: if model does not exist. Use AlloyDBModelManager to create the model.

        """
        self._engine = engine
        self.model_id = model_id

        self.model_manager = AlloyDBModelManager(engine=self._engine)
        if not self.model_exists():
            raise ValueError(f"Model {model_id} does not exist.")

    async def amodel_exists(self) -> bool:
        """Checks if the embedding model exists.

        Return:
            `Bool`: True if a model with the given name exists, False otherwise.
        """
        return await self._engine._run_as_async(self.__amodel_exists())

    def model_exists(self) -> bool:
        """Checks if the embedding model exists.

        Return:
            `Bool`: True if a model with the given name exists, False otherwise.
        """
        return self._engine._run_as_sync(self.__amodel_exists())

    async def __amodel_exists(self) -> bool:
        """Checks if the embedding model exists.

        Return:
            `Bool`: True if a model with the given name exists, False otherwise.
        """
        model = await self.model_manager.aget_model(model_id=self.model_id)
        if model is not None:
            return True
        return False

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
        embeddings = await self._engine._run_as_async(self.__aembed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            query (str): Text to embed.

        Returns:
            List[float]: Embedding.
        """
        return self._engine._run_as_sync(self.__aembed_query(text))

    async def __aembed_query(self, query: str) -> List[float]:
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
