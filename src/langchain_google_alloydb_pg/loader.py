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

from __future__ import annotations

from typing import AsyncIterator, Callable, Iterator, Optional

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .async_loader import AsyncAlloyDBDocumentSaver, AsyncAlloyDBLoader
from .engine import AlloyDBEngine

DEFAULT_CONTENT_COL = "page_content"
DEFAULT_METADATA_COL = "langchain_metadata"


class AlloyDBLoader(BaseLoader):
    """Load documents from AlloyDB`.

    Each document represents one row of the result. The `content_columns` are
    written into the `content_columns` of the document. The `metadata_columns` are written
    into the `metadata_columns` of the document. By default, first columns is written into
    the `page_content` and everything else into the `metadata`.
    """

    __create_key = object()

    def __init__(
        self, key: object, engine: AlloyDBEngine, loader: AsyncAlloyDBLoader
    ) -> None:
        """AlloyDBLoader constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): AsyncEngine with pool connection to the postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".


        Raises:
            Exception: If called directly by user.
        """
        if key != AlloyDBLoader.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self._engine = engine
        self.__loader = loader

    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
    ) -> AlloyDBLoader:
        """Create a new AlloyDBLoader instance.

        Args:
            engine (AlloyDBEngine):AsyncEngine with pool connection to the postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.

        Returns:
            AlloyDBLoader
        """
        coro = AsyncAlloyDBLoader.create(
            engine,
            query,
            table_name,
            schema_name,
            content_columns,
            metadata_columns,
            metadata_json_column,
            format,
            formatter,
        )
        loader = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, loader)

    @classmethod
    def create_sync(
        cls,
        engine: AlloyDBEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
    ) -> AlloyDBLoader:
        """Create a new AlloyDBLoader instance.

        Args:
            engine (AlloyDBEngine):AsyncEngine with pool connection to the postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.

        Returns:
            AlloyDBLoader
        """
        coro = AsyncAlloyDBLoader.create(
            engine,
            query,
            table_name,
            schema_name,
            content_columns,
            metadata_columns,
            metadata_json_column,
            format,
            formatter,
        )
        loader = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, loader)

    def load(self) -> list[Document]:
        """Load PostgreSQL data into Document objects."""
        return self._engine._run_as_sync(self.__loader.aload())

    async def aload(self) -> list[Document]:
        """Load PostgreSQL data into Document objects."""
        return await self._engine._run_as_async(self.__loader.aload())

    def lazy_load(self) -> Iterator[Document]:
        """Load PostgreSQL data into Document objects lazily."""
        iterator = self.__loader.alazy_load()
        while True:
            try:
                result = self._engine._run_as_sync(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load PostgreSQL data into Document objects lazily."""
        iterator = self.__loader.alazy_load()
        while True:
            try:
                result = await self._engine._run_as_async(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break


class AlloyDBDocumentSaver:
    """A class for saving langchain documents into a PostgreSQL database table."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
        saver: AsyncAlloyDBDocumentSaver,
    ):
        """AlloyDBDocumentSaver constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): AsyncEngine with pool connection to the postgres database
            table_name (str): Name of table to query.
            content_column (str): Column that represent a Document's page_content.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            metadata_columns (list[str], optional): Column(s) that represent a Document's metadata. Defaults to empty list.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to None.

        Raises:
            Exception: if called directly by user.
        """
        if key != AlloyDBDocumentSaver.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self._saver = saver

    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        content_column: str = DEFAULT_CONTENT_COL,
        metadata_columns: list[str] = [],
        metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
    ) -> AlloyDBDocumentSaver:
        """Create an AlloyDBDocumentSaver instance.

        Args:
            engine (AlloyDBEngine):AsyncEngine with pool connection to the postgres database
            table_name (str): Name of table to query.
            schema_name (str, optional): Name of schema where the table is located. Defaults to "public".
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "page_content".
            metadata_columns (list[str], optional): Column(s) that represent a Document's metadata. Defaults to an empty list.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            AlloyDBDocumentSaver
        """
        coro = AsyncAlloyDBDocumentSaver.create(
            engine,
            table_name,
            schema_name,
            content_column,
            metadata_columns,
            metadata_json_column,
        )
        saver = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, saver)

    @classmethod
    def create_sync(
        cls,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        content_column: str = DEFAULT_CONTENT_COL,
        metadata_columns: list[str] = [],
        metadata_json_column: str = DEFAULT_METADATA_COL,
    ) -> AlloyDBDocumentSaver:
        """Create an AlloyDBDocumentSaver instance.

        Args:
            engine (AlloyDBEngine):AsyncEngine with pool connection to the postgres database
            table_name (str): Name of table to query.
            schema_name (str, optional): Name of schema where the table is located. Defaults to "public".
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "page_content".
            metadata_columns (list[str], optional): Column(s) that represent a Document's metadata. Defaults to an empty list.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            AlloyDBDocumentSaver
        """
        coro = AsyncAlloyDBDocumentSaver.create(
            engine,
            table_name,
            schema_name,
            content_column,
            metadata_columns,
            metadata_json_column,
        )
        saver = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, saver)

    async def aadd_documents(self, docs: list[Document]) -> None:
        """
        Save documents in the DocumentSaver table. Document’s metadata is added to columns if found or
        stored in langchain_metadata JSON column.

        Args:
            docs (list[langchain_core.documents.Document]): List of documents to be saved.
        """
        await self._engine._run_as_async(self._saver.aadd_documents(docs))

    def add_documents(self, docs: list[Document]) -> None:
        """
        Save documents in the DocumentSaver table. Document’s metadata is added to columns if found or
        stored in langchain_metadata JSON column.

        Args:
            docs (list[langchain_core.documents.Document]): List of documents to be saved.
        """
        self._engine._run_as_sync(self._saver.aadd_documents(docs))

    async def adelete(self, docs: list[Document]) -> None:
        """
        Delete all instances of a document from the DocumentSaver table by matching the entire Document
        object.

        Args:
            docs (list[langchain_core.documents.Document]): List of documents to be deleted.
        """
        await self._engine._run_as_async(self._saver.adelete(docs))

    def delete(self, docs: list[Document]) -> None:
        """
        Delete all instances of a document from the DocumentSaver table by matching the entire Document
        object.

        Args:
            docs (list[langchain_core.documents.Document]): List of documents to be deleted.
        """
        self._engine._run_as_sync(self._saver.adelete(docs))
