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

from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from .engine import AlloyDBEngine
from .model_manager import AlloyDBModelManager


class AlloyDBEmbeddings(Embeddings):
    """Google AlloyDB Embeddings available via Model Endpoint Management."""

    __create_key = object()

    def __init__(self, key: object, engine: AlloyDBEngine, model_id: str):
        """AlloyDBEmbeddings constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        Raises:
            :class:`ValueError`: if model does not exist. Use AlloyDBModelManager to create the model.

        """
        if key != AlloyDBEmbeddings.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.model_id = model_id

    @classmethod
    async def create(
        cls: type[AlloyDBEmbeddings], engine: AlloyDBEngine, model_id: str
    ) -> AlloyDBEmbeddings:
        """Create AlloyDBEmbeddings instance.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        Returns:
            AlloyDBEmbeddings: Instance of AlloyDBEmbeddings.
        """

        embeddings = cls(cls.__create_key, engine, model_id)
        model_exists = await embeddings.amodel_exists()
        if not model_exists:
            raise ValueError(f"Model {model_id} does not exist.")

        return embeddings

    @classmethod
    def create_sync(
        cls: type[AlloyDBEmbeddings], engine: AlloyDBEngine, model_id: str
    ) -> AlloyDBEmbeddings:
        """Create AlloyDBEmbeddings instance.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            model_id (str): The model id used for generating embeddings.

        Returns:
            AlloyDBEmbeddings: Instance of AlloyDBEmbeddings.
        """

        embeddings = cls(cls.__create_key, engine, model_id)
        if not embeddings.model_exists():
            raise ValueError(f"Model {model_id} does not exist.")

        return embeddings

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
        model_manager = await AlloyDBModelManager.create(self._engine)
        model = await model_manager.aget_model(model_id=self.model_id)
        if model is not None:
            return True
        return False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Embedding functions are not implemented. Use VertexAIEmbeddings interface instead."
        )

    def embed_query_inline(self, query: str) -> str:
        return f"embedding('{self.model_id}', '{query}')::vector"

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            query (str): Text to embed.

        Returns:
            list[float]: Embedding.
        """
        embeddings = await self._engine._run_as_async(self.__aembed_query(text))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            query (str): Text to embed.

        Returns:
            list[float]: Embedding.
        """
        return self._engine._run_as_sync(self.__aembed_query(text))

    async def __aembed_query(self, query: str) -> list[float]:
        """Coroutine for generating embeddings for a given query.

        Args:
            query (str): Text to embed.

        Returns:
            list[float]: Embedding.
        """
        query = f" SELECT embedding('{self.model_id}', '{query}')::vector "
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return json.loads(results[0]["embedding"])
