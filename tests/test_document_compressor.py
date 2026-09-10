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

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_google_alloydb_pg.document_compressor import (
    AlloyDBDocumentCompressor,
)
from langchain_google_alloydb_pg.engine import AlloyDBEngine


@pytest.fixture
def mock_engine():
    """Fixture providing a mocked AlloyDBEngine with async pool context."""

    class DummyEngine(AlloyDBEngine):
        def __init__(self):
            pass

        _pool = MagicMock()
        _run_as_sync = MagicMock()

    engine = DummyEngine()
    engine._pool = MagicMock()
    return engine


@pytest.fixture
def mock_documents():
    """Fixture providing sample test documents."""
    return [
        Document(page_content="The quick brown fox."),
        Document(page_content="Jumps over the lazy dog."),
    ]


@pytest.mark.asyncio
async def test_compressor_arun_score_per_row(mock_engine, mock_documents):
    """Test async document compression with single-score row returns from google_ml.rank."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[0.9], [0.1]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(
        engine=mock_engine,
        model_id="semantic-ranker-512@latest",
        top_n=2,
    )
    result = await compressor.acompress_documents(mock_documents, "query about a fox")

    assert len(result) == 2
    assert result[0].page_content == "The quick brown fox."
    assert result[0].metadata["relevance_score"] == 0.9
    assert result[1].page_content == "Jumps over the lazy dog."
    assert result[1].metadata["relevance_score"] == 0.1

    # Verify parameterized SQL execution and query text
    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT * FROM google_ml.rank" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params["model_id"] == "semantic-ranker-512@latest"
    assert executed_params["query"] == "query about a fox"
    assert executed_params["documents"] == [
        "The quick brown fox.",
        "Jumps over the lazy dog.",
    ]
    assert executed_params["top_n"] == 2


@pytest.mark.asyncio
async def test_compressor_arun_table_indexed_reranking(mock_engine):
    """Test async document compression with 1-based (index, score) table response."""
    docs = [
        Document(page_content="Document 1 - lower relevance."),
        Document(page_content="Document 2 - higher relevance."),
        Document(page_content="Document 3 - medium relevance."),
    ]
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # AlloyDB google_ml.rank returns TABLE(index integer, score real), 1-indexed
    exec_result.fetchall.return_value = [[2, 0.95], [3, 0.70], [1, 0.30]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(
        engine=mock_engine,
        top_n=2,
    )
    result = await compressor.acompress_documents(docs, "find relevant docs")

    assert len(result) == 2
    # First returned should be Document 2 (index 2 in Postgres -> index 1 in Python)
    assert result[0].page_content == "Document 2 - higher relevance."
    assert result[0].metadata["relevance_score"] == 0.95
    # Second returned should be Document 3 (index 3 in Postgres -> index 2 in Python)
    assert result[1].page_content == "Document 3 - medium relevance."
    assert result[1].metadata["relevance_score"] == 0.70


@pytest.mark.asyncio
async def test_compressor_arun_array_of_scores(mock_engine, mock_documents):
    """Test async document compression with single row returning score array."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # Single row containing array of scores: [score_for_doc_0, score_for_doc_1]
    exec_result.fetchall.return_value = [[[0.2, 0.85]]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(
        engine=mock_engine,
        top_n=1,
    )
    result = await compressor.acompress_documents(mock_documents, "query")

    assert len(result) == 1
    # Sorted descending by score, so doc 1 (score 0.85) should be first
    assert result[0].page_content == "Jumps over the lazy dog."
    assert result[0].metadata["relevance_score"] == 0.85


@pytest.mark.asyncio
async def test_compressor_arun_empty_documents(mock_engine):
    """Test that passing an empty document sequence short-circuits without DB query."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents([], "query")
    assert result == []
    assert not mock_engine._pool.connect.called


@pytest.mark.asyncio
async def test_compressor_arun_index_error_fallback(mock_engine, mock_documents):
    """Test fallback when row tuple index is unparseable or out of bounds."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # Invalid index string "invalid_idx" triggers ValueError/IndexError fallback
    exec_result.fetchall.return_value = [["invalid_idx", 0.75], [999, 0.25]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(mock_documents, "query")

    assert len(result) == 2
    assert result[0].page_content == "The quick brown fox."
    assert result[1].page_content == "Jumps over the lazy dog."


def test_compressor_run_sync(mock_engine, mock_documents):
    """Test synchronous compress_documents delegates to _run_as_sync with 0 warnings."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    mock_docs_ranked = [
        Document(
            page_content="The quick brown fox.",
            metadata={"relevance_score": 0.9},
        ),
    ]

    def mock_run_sync(coro):
        # Explicitly close the unawaited coroutine in mock side effect to prevent RuntimeWarning
        coro.close()
        return mock_docs_ranked

    with patch.object(mock_engine, "_run_as_sync", side_effect=mock_run_sync):
        result = compressor.compress_documents(mock_documents, "query about a fox")
        assert len(result) == 1
        assert result[0].page_content == "The quick brown fox."
        assert result[0].metadata["relevance_score"] == 0.9


def test_compressor_default_attributes(mock_engine):
    """Test default values of model_id and top_n."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    assert compressor.model_id == "semantic-ranker-512@latest"
    assert compressor.top_n is None
