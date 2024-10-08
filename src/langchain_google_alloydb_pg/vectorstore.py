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

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .async_vectorstore import AsyncAlloyDBVectorStore
from .engine import AlloyDBEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BaseIndex,
    DistanceStrategy,
    QueryOptions,
)


class AlloyDBVectorStore(VectorStore):
    """Google AlloyDB Vector Store class"""

    __create_key = object()

    def __init__(self, key: object, engine: AlloyDBEngine, vs: AsyncAlloyDBVectorStore):
        """AlloyDBVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
            vs (AsyncAlloyDBVectorstore): The async only VectorStore implementation


        Raises:
            Exception: If called directly by user.
        """
        if key != AlloyDBVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self._engine = engine
        self.__vs = vs

    @classmethod
    async def create(
        cls: Type[AlloyDBVectorStore],
        engine: AlloyDBEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance.

        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Returns:
            AlloyDBVectorStore
        """
        coro = AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
        )
        vs = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, vs)

    @classmethod
    def create_sync(
        cls,
        engine: AlloyDBEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            content_column (str, optional): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[List[str]]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy, optional): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int, optional): Number of Documents to return from search. Defaults to 4.
            fetch_k (int, optional): Number of Documents to fetch to pass to MMR algorithm. Defaults to 20.
            lambda_mult (float, optional): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (Optional[QueryOptions], optional): Index query option. Defaults to None.

        Returns:
            AlloyDBVectorStore
        """
        coro = AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
        )
        vs = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, vs)

    @property
    def embeddings(self) -> Embeddings:
        return self.__vs.embedding_service

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add data along with embeddings to the table."""
        return await self._engine._run_as_async(
            self.__vs.aadd_embeddings(texts, embeddings, metadatas, ids, **kwargs)
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(
            self.__vs.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed documents and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(
            self.__vs.aadd_documents(documents, ids, **kwargs)
        )

    async def aadd_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed images and add to the table."""
        return await self._engine._run_as_async(
            self.__vs.aadd_images(uris, metadatas, ids, **kwargs)
        )

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add data along with embeddings to the table."""
        return self._engine._run_as_sync(
            self.__vs.aadd_embeddings(texts, embeddings, metadatas, ids, **kwargs)
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(
            self.__vs.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed documents and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(
            self.__vs.aadd_documents(documents, ids, **kwargs)
        )

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed images and add to the table."""
        return self._engine._run_as_sync(
            self.__vs.aadd_images(uris, metadatas, ids, **kwargs)
        )

    async def adelete(
        self,
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(self.__vs.adelete(ids, **kwargs))

    def delete(
        self,
        ids: Optional[List] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(self.__vs.adelete(ids, **kwargs))

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls: Type[AlloyDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance from texts.

        Args:
            texts (List[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            metadatas (Optional[List[dict]], optional): List of metadatas to add to table records. Defaults to None.
            ids: (Optional[List]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[List[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AlloyDBVectorStore
        """
        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
        )
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    async def afrom_documents(  # type: ignore[override]
        cls: Type[AlloyDBVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[List] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance from documents.

        Args:
            documents (List[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            ids: (Optional[List]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[List[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AlloyDBVectorStore
        """

        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
        )
        await vs.aadd_documents(documents, ids=ids)
        return vs

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[AlloyDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance from texts.

        Args:
            texts (List[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            metadatas (Optional[List[dict]], optional): List of metadatas to add to table records. Defaults to None.
            ids: (Optional[List]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str], optional): Column(s) that represent a document's metadata. Defaults to empty list.
            ignore_metadata_columns (Optional[List[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AlloyDBVectorStore
        """
        vs = cls.create_sync(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            **kwargs,
        )
        vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: Type[AlloyDBVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[List] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> AlloyDBVectorStore:
        """Create an AlloyDBVectorStore instance from documents.

        Args:
            documents (List[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (AlloyDBEngine): Connection pool engine for managing connections to AlloyDB database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            ids: (Optional[List]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[List[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AlloyDBVectorStore
        """
        vs = cls.create_sync(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            **kwargs,
        )
        vs.add_documents(documents, ids=ids)
        return vs

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on query."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search(query, k, filter, **kwargs)
        )

    def similarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on image."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_image(image_uri, k, filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on query."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search(query, k, filter, **kwargs)
        )

    async def asimilarity_search_image(
        self,
        image_uri: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on query."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search(image_uri, k, filter, **kwargs)
        )

    # Required for (a)similarity_search_with_relevance_scores
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select a relevance function based on distance strategy."""
        # Calculate distance strategy provided in vectorstore constructor
        if self.__vs.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        if self.__vs.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.__vs.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by vector similarity search."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_by_vector(embedding, k, filter, **kwargs)
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by vector similarity search."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_with_score_by_vector(
                embedding, k, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search_with_score_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by vector similarity search."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_by_vector(embedding, k, filter, **kwargs)
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on vector."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_with_score_by_vector(
                embedding, k, filter, **kwargs
            )
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search_with_score_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
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

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return await self._engine._run_as_async(
            self.__vs.aapply_vector_index(index, name, concurrently)
        )

    def apply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return self._engine._run_as_sync(
            self.__vs.aapply_vector_index(index, name, concurrently)
        )

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return await self._engine._run_as_async(self.__vs.areindex(index_name))

    def reindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return self._engine._run_as_sync(self.__vs.areindex(index_name))

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return await self._engine._run_as_async(
            self.__vs.adrop_vector_index(index_name)
        )

    def drop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return self._engine._run_as_sync(self.__vs.adrop_vector_index(index_name))

    async def ais_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return await self._engine._run_as_async(self.__vs.is_valid_index(index_name))

    def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return self._engine._run_as_sync(self.__vs.is_valid_index(index_name))

    def get_table_name(self) -> str:
        return self.__vs.table_name
