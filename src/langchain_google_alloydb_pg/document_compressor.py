# Copyright 2026 Google LLC
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

import asyncio
from typing import Any, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, field_validator
from sqlalchemy import text

from .engine import AlloyDBEngine

# The parameters are cast explicitly because google_ml.rank is overloaded and
# untyped bind parameters make the call ambiguous (SQLSTATE 42725).
RANK_QUERY = """
    SELECT index, score FROM google_ml.rank(
        CAST(:model_id AS VARCHAR),
        CAST(:query AS TEXT),
        CAST(:documents AS TEXT[]),
        CAST(:top_n AS INTEGER)
    )
"""


def _to_ranked_documents(
    documents: Sequence[Document], rows: Sequence[Sequence[Any]]
) -> list[Document]:
    """Map ``(index, score)`` rows from google_ml.rank back to documents.

    ``index`` is the 1-based position of the document in the input array.
    Rows with a NULL score or an index outside the input are skipped. The
    input documents are not modified; each result is a copy whose metadata
    includes ``relevance_score``. Results are sorted by score, highest first.
    """
    ranked = []
    for index, score in rows:
        if score is None or not 1 <= index <= len(documents):
            continue
        doc = documents[index - 1]
        ranked.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": float(score)},
            )
        )
    ranked.sort(key=lambda d: d.metadata["relevance_score"], reverse=True)
    return ranked


class AlloyDBDocumentCompressor(BaseDocumentCompressor):
    """Reranks documents with AlloyDB AI's ``google_ml.rank()`` function.

    The ranking runs inside the database through AlloyDB's Vertex AI
    integration. ``model_id`` is required, for example
    ``"semantic-ranker-default-003"``.

    .. code-block:: python

        compressor = AlloyDBDocumentCompressor(
            engine=engine, model_id="semantic-ranker-default-003", top_n=3
        )
        docs = compressor.compress_documents(documents, "What is AlloyDB?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: AlloyDBEngine
    model_id: str
    top_n: Optional[int] = None

    @field_validator("model_id")
    @classmethod
    def check_model_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must be a non-empty string.")
        return v

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Rerank documents by relevance to the query.

        Args:
            documents (Sequence[Document]): Documents to rerank.
            query (str): The query to rank the documents against.
            callbacks (Optional[Callbacks]): Unused. Defaults to None.

        Returns:
            Sequence[Document]: Copies of the documents, sorted by
            ``relevance_score`` (highest first) and limited to ``top_n``.

        Raises:
            RuntimeError: If called from the engine's background event loop.
        """
        try:
            running_loop: Optional[asyncio.AbstractEventLoop] = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop is self.engine._loop:
            raise RuntimeError(
                "Cannot call synchronous 'compress_documents' from the engine's "
                "event loop because it would deadlock. Use "
                "'acompress_documents' instead."
            )
        return self.engine._run_as_sync(self.__acompress_documents(documents, query))

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Rerank documents by relevance to the query.

        Args:
            documents (Sequence[Document]): Documents to rerank.
            query (str): The query to rank the documents against.
            callbacks (Optional[Callbacks]): Unused. Defaults to None.

        Returns:
            Sequence[Document]: Copies of the documents, sorted by
            ``relevance_score`` (highest first) and limited to ``top_n``.
        """
        return await self.engine._run_as_async(
            self.__acompress_documents(documents, query)
        )

    async def __acompress_documents(
        self, documents: Sequence[Document], query: str
    ) -> list[Document]:
        """Validate the input, call google_ml.rank and map the rows back."""
        if not documents:
            return []
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be a positive integer greater than 0.")

        params = {
            "model_id": self.model_id,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": self.top_n if self.top_n is not None else len(documents),
        }
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(RANK_QUERY), params)
            rows = result.fetchall()

        return _to_ranked_documents(documents, rows)[: self.top_n]
