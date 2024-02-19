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

import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    cast,
)

import sqlalchemy
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .alloydb_engine import AlloyDBEngine

DEFAULT_CONTENT_COL = "page_content"
DEFAULT_METADATA_COL = "langchain_metadata"
DEFAULT_FORMAT = "text"


def text_formatter(row, content_columns) -> str:
    return " ".join(str(row[column]) for column in content_columns if column in row)


def cvs_formatter(row, content_columns) -> str:
    return ", ".join(str(row[column]) for column in content_columns if column in row)


def yaml_formatter(row, content_columns) -> str:
    return "\n".join(
        f"{column}: {str(row[column])}" for column in content_columns if column in row
    )


def json_formatter(row, content_columns) -> str:
    dictionary = {}
    for column in content_columns:
        if column in row:
            dictionary[column] = row[column]
    return json.dumps(dictionary)


def _parse_doc_from_row(
    content_columns: Iterable[str],
    metadata_columns: Iterable[str],
    row: dict,
    metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
    formatter: Callable = text_formatter,
) -> Document:
    page_content = formatter(row, content_columns)
    metadata: Dict[str, Any] = {}
    # unnest metadata from langchain_metadata column
    if metadata_json_column and row.get(metadata_json_column):
        for k, v in row[metadata_json_column].items():
            metadata[k] = v
    # load metadata from other columns
    for column in metadata_columns:
        if column in row and column != metadata_json_column:
            metadata[column] = row[column]

    return Document(page_content=page_content, metadata=metadata)


class AlloyDBLoader(BaseLoader):
    """Load documents from Alloydb`.

    Each document represents one row of the result. The `content_columns` are
    written into the `content_columns`of the document. The `metadata_columns` are written
    into the `metadata_columns` of the document. By default, first columns is written into
    the `page_content` and everything else into the `metadata`.
    """

    def __init__(
        self,
        engine: AlloyDBEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
        read_only: Optional[bool] = None,
        time_out: Optional[int] = None,
        metadata_json_column: Optional[str] = None,
    ) -> None:
        """Initialize Alloydb document loader."""

        self.engine = engine
        self.table_name = table_name
        self.query = query
        if table_name and query:
            raise ValueError("Only one of 'table_name' or 'query' should be specified.")
        if not table_name and not query:
            raise ValueError(
                "At least one of the parameters 'table_name' or 'query' needs to be provided"
            )
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        if format and formatter:
            raise ValueError("Only one of 'format' or 'formatter' should be specified.")

        if format and format not in ["csv", "text", "JSON", "YAML"]:
            raise ValueError("format must be type: 'csv', 'text', 'JSON', 'YAML'")
        self.format = format or DEFAULT_FORMAT
        self.formatter = formatter
        self.read_only = read_only
        self.time_out = time_out
        self.metadata_json_column = metadata_json_column

    async def _collect_async_items(self, docs_generator):
        return [doc async for doc in docs_generator]

    def load(self) -> List[Document]:
        """Load Alloydb data into Document objects."""
        documents = self.engine.run_as_sync(
            self._collect_async_items(self.alazy_load())
        )
        return documents

    async def aload(self) -> List[Document]:
        return [doc async for doc in self.alazy_load()]

    def lazy_load(self) -> Iterator[Document]:
        yield from self.engine.run_as_sync(self._collect_async_items(self.alazy_load()))

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load Alloydb data into Document objects lazily."""
        content_columns = self.content_columns
        metadata_columns = self.metadata_columns

        if self.query:
            stmt = sqlalchemy.text(self.query)
        else:
            stmt = sqlalchemy.text(f'SELECT * FROM "{self.table_name}"')

        async with self.engine._engine.connect() as connection:
            result_proxy = await connection.execute(stmt)
            # Get field type information.
            # cursor.description is a sequence of 7-item sequences.
            # Each of these sequences contains information describing one result column:
            # - name, type_code, display_size, internal_size, precision, scale, null_ok
            # The first two items (name and type_code) are mandatory, the other five are optional
            # and are set to None if no meaningful values can be provided.
            # link: https://peps.python.org/pep-0249/#description
            # column_types = [
            #     cast(tuple, field)[0:2]
            #     for field in result_proxy.cursor.description
            # ]
            column_names = list(result_proxy.keys())
            content_columns = self.content_columns or [column_names[0]]
            metadata_columns = self.metadata_columns or [
                col for col in column_names if col not in content_columns
            ]

            # check validity of metadata json column
            if (
                self.metadata_json_column
                and self.metadata_json_column not in column_names
            ):
                raise ValueError(
                    f"Column {self.metadata_json_column} not found in query result {column_names}."
                )
            # use default metadata json column if not specified
            if self.metadata_json_column and self.metadata_json_column in column_names:
                metadata_json_column = self.metadata_json_column
            elif DEFAULT_METADATA_COL in column_names:
                metadata_json_column = DEFAULT_METADATA_COL
            else:
                metadata_json_column = None
            # check validity of other column
            all_names = content_columns + metadata_columns
            for name in all_names:
                if name not in column_names:
                    raise ValueError(
                        f"Column {name} not found in query result {column_names}."
                    )
            # load document one by one
            while True:
                row = result_proxy.fetchone()
                if not row:
                    break
                row_data = {}
                for column in column_names:
                    value = getattr(row, column)
                    row_data[column] = value
                if self.formatter:
                    formatter = self.formatter
                elif self.format == "csv":
                    formatter = cvs_formatter
                elif self.format == "YAML":
                    formatter = yaml_formatter
                elif self.format == "JSON":
                    formatter = json_formatter
                else:
                    formatter = text_formatter

                yield _parse_doc_from_row(
                    content_columns,
                    metadata_columns,
                    row_data,
                    metadata_json_column,
                    formatter,
                )
