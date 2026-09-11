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
from typing import Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from sqlalchemy import text

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None  # type: ignore[assignment,misc]

from .engine import AlloyDBEngine


class AlloyDBDocumentCompressor(BaseDocumentCompressor):
    """Document Compressor that uses AlloyDB's google_ml.rank() for reranking.

    This class leverages AlloyDB's native Vertex AI ranking integration to rerank documents
    based on a query directly in the database.
    """

    engine: AlloyDBEngine
    model_id: str = "semantic-ranker-512@latest"
    top_n: Optional[int] = None

    if hasattr(BaseDocumentCompressor, "model_config") and ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using AlloyDB's rank model."""
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if (
            curr_loop is not None
            and getattr(self.engine, "_loop", None) is not None
            and curr_loop == self.engine._loop
        ):
            raise RuntimeError(
                "Cannot call synchronous 'compress_documents' from the engine's background event loop "
                "as it causes a thread deadlock. Use 'acompress_documents' instead."
            )
        return self.engine._run_as_sync(
            self.acompress_documents(documents, query, callbacks)
        )

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Asynchronously compress documents using AlloyDB's rank model."""
        if not documents:
            return []

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be a positive integer greater than 0.")

        texts = [doc.page_content for doc in documents]

        # Parameterized query calling google_ml.rank
        query_text = """
            SELECT * FROM google_ml.rank(:model_id, :query, :documents, :top_n)
        """

        async def _query():
            async with self.engine._pool.connect() as conn:
                result = await conn.execute(
                    text(query_text),
                    {
                        "model_id": self.model_id,
                        "query": query,
                        "documents": texts,
                        "top_n": (
                            self.top_n if self.top_n is not None else len(documents)
                        ),
                    },
                )
                return result.fetchall()

        rows = await self.engine._run_as_async(_query())

        compressed_docs = []
        # Support fallback to array of scores if returned by custom transforms
        if len(rows) > 0 and len(rows[0]) == 1 and isinstance(rows[0][0], list):
            scores = rows[0][0]
            for idx, score in enumerate(scores):
                if idx >= len(documents):
                    break
                if score is None:
                    continue
                orig_doc = documents[idx]
                new_metadata = dict(orig_doc.metadata)
                try:
                    score_float = float(score)
                except (ValueError, TypeError, OverflowError):
                    continue
                new_metadata["relevance_score"] = score_float
                compressed_docs.append(
                    Document(page_content=orig_doc.page_content, metadata=new_metadata)
                )
            compressed_docs.sort(
                key=lambda x: x.metadata["relevance_score"], reverse=True
            )
            if self.top_n is not None:
                compressed_docs = compressed_docs[: self.top_n]
        else:
            # Table-valued return: TABLE(index integer, score real)
            for row in rows:
                if len(row) >= 2:
                    try:
                        raw_idx, raw_score = row[0], row[1]
                        if raw_score is None:
                            continue
                        if isinstance(raw_idx, float) and not raw_idx.is_integer():
                            continue
                        raw_idx_int = int(raw_idx)
                        if raw_idx_int < 1 or raw_idx_int > len(documents):
                            raise IndexError(
                                f"Index {raw_idx_int} out of 1-based bounds [1, {len(documents)}]"
                            )
                        doc_idx = raw_idx_int - 1
                        orig_doc = documents[doc_idx]
                        score = float(raw_score)
                        new_metadata = dict(orig_doc.metadata)
                        new_metadata["relevance_score"] = score
                        compressed_docs.append(
                            Document(
                                page_content=orig_doc.page_content,
                                metadata=new_metadata,
                            )
                        )
                    except (ValueError, TypeError, IndexError, OverflowError):
                        continue
                else:
                    # Single column returns without document index cannot be safely mapped
                    continue

        if self.top_n is not None:
            compressed_docs = compressed_docs[: self.top_n]

        return compressed_docs
