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

from typing import Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from sqlalchemy import text

from .engine import AlloyDBEngine


class AlloyDBDocumentCompressor(BaseDocumentCompressor):
    """Document Compressor that uses AlloyDB's google_ml.rank() for reranking.
    
    This class leverages AlloyDB's native Vertex AI ranking integration to rerank documents
    based on a query directly in the database.
    """
    
    engine: AlloyDBEngine
    model_id: str = "semantic-ranker-512@latest"
    top_n: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using AlloyDB's rank model."""
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
            
        texts = [doc.page_content for doc in documents]
        
        # We construct a parameterized query that calls google_ml.rank
        # The return type depends on the exact function signature, but typically it returns rows
        # with an index or id mapping back to the input array.
        # We query the function and rely on the order it returns to sort the documents.
        query_text = """
            SELECT * FROM google_ml.rank(:model_id, :query, :documents, :top_n)
        """
        
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(
                text(query_text),
                {
                    "model_id": self.model_id,
                    "query": query,
                    "documents": texts,
                    "top_n": self.top_n if self.top_n is not None else len(documents)
                }
            )
            rows = result.fetchall()
            
        compressed_docs = []
        # Fallback to standard 1-to-1 if returns raw scores array
        if len(rows) > 0 and len(rows[0]) == 1 and isinstance(rows[0][0], list):
            # It returned a single row with an array of scores
            scores = rows[0][0]
            for idx, score in enumerate(scores):
                if self.top_n and idx >= self.top_n:
                    continue
                doc = documents[idx]
                doc.metadata["relevance_score"] = float(score)
                compressed_docs.append(doc)
            # Sort by score descending
            compressed_docs.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)
            if self.top_n:
                compressed_docs = compressed_docs[:self.top_n]
        else:
            # It returned rows, hopefully with (index, score) or just scores
            for idx, row in enumerate(rows):
                if len(row) >= 2:
                    # Assuming (index, score) or (id, score)
                    # We will just map it positionally for now or try to extract index
                    try:
                        doc_idx = int(row[0]) - 1 # Postgres arrays are 1-indexed
                        doc = documents[doc_idx]
                        doc.metadata["relevance_score"] = float(row[1])
                    except (ValueError, TypeError, IndexError):
                        doc = documents[idx]
                        doc.metadata["relevance_score"] = float(row[0])
                else:
                    doc = documents[idx]
                    doc.metadata["relevance_score"] = float(row[0])
                compressed_docs.append(doc)
            
        return compressed_docs
