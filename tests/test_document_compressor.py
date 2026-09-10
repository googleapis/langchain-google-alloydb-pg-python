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
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from langchain_google_alloydb_pg.document_compressor import (
    AlloyDBDocumentCompressor,
)
from langchain_google_alloydb_pg.engine import AlloyDBEngine


@pytest.fixture
def mock_engine():
    """Fixture providing a mock AlloyDBEngine with pool and sync/async runners."""

    class DummyEngine(AlloyDBEngine):

        def __init__(self):
            pass

    engine = DummyEngine()
    pool_mock = MagicMock()
    conn_mock = AsyncMock()
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[1, 0.95]]
    conn_mock.execute.return_value = exec_result
    pool_mock.connect.return_value.__aenter__.return_value = conn_mock
    engine._pool = pool_mock

    async def _real_async_runner(coro):
        return await coro

    engine._run_as_async = MagicMock(side_effect=_real_async_runner)

    def _real_sync_runner(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    engine._run_as_sync = MagicMock(side_effect=_real_sync_runner)
    return engine


@pytest.fixture
def sample_documents():
    """Fixture providing a set of test documents."""
    return [
        Document(
            page_content="Alpha: Introduction to Machine Learning.",
            metadata={"source": "book_a", "page": 1},
        ),
        Document(
            page_content="Beta: Deep Learning and Neural Networks.",
            metadata={"source": "book_b", "page": 42},
        ),
        Document(
            page_content="Gamma: Advanced Database Systems in Cloud.",
            metadata={"source": "book_c", "page": 100},
        ),
    ]


def test_compressor_default_attributes(mock_engine):
    """Test default values of model_id and top_n."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    assert compressor.model_id == "semantic-ranker-512@latest"
    assert compressor.top_n is None


def test_compressor_custom_attributes(mock_engine):
    """Test custom configuration of model_id and top_n."""
    compressor = AlloyDBDocumentCompressor(
        engine=mock_engine,
        model_id="custom-ranker",
        top_n=5,
    )
    assert compressor.model_id == "custom-ranker"
    assert compressor.top_n == 5


@pytest.mark.asyncio
async def test_compress_documents_empty_list_async(mock_engine):
    """Test that passing an empty document sequence short-circuits without DB query."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents([], "query")
    assert result == []
    assert not mock_engine._run_as_async.called
    assert not mock_engine._pool.connect.called


def test_compress_documents_empty_list_sync(mock_engine):
    """Test synchronous compress_documents with empty document list returns empty."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = compressor.compress_documents([], "query")
    assert result == []
    assert not mock_engine._pool.connect.called


@pytest.mark.asyncio
async def test_compress_documents_happy_path_async(mock_engine, sample_documents):
    """Test standard async document compression and ranking with AlloyDB."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # 1-based table output: (index, score)
    exec_result.fetchall.return_value = [[2, 0.95], [1, 0.80], [3, 0.40]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(
        engine=mock_engine,
        model_id="semantic-ranker-512@latest",
        top_n=3,
    )
    result = await compressor.acompress_documents(sample_documents, "machine learning")

    assert len(result) == 3
    # Document 2 (index 2 -> sample_documents[1]) ranked highest
    assert result[0].page_content == sample_documents[1].page_content
    assert result[0].metadata["relevance_score"] == 0.95

    # Document 1 (index 1 -> sample_documents[0]) ranked second
    assert result[1].page_content == sample_documents[0].page_content
    assert result[1].metadata["relevance_score"] == 0.80

    # Document 3 (index 3 -> sample_documents[2]) ranked third
    assert result[2].page_content == sample_documents[2].page_content
    assert result[2].metadata["relevance_score"] == 0.40

    # Verify query and parameters
    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT * FROM google_ml.rank" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params["model_id"] == "semantic-ranker-512@latest"
    assert executed_params["query"] == "machine learning"
    assert executed_params["top_n"] == 3
    assert len(executed_params["documents"]) == 3


def test_compress_documents_happy_path_sync(mock_engine, sample_documents):
    """Test synchronous compress_documents delegates to engine._run_as_sync and executes pipeline."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # 1-based table output: (index, score)
    exec_result.fetchall.return_value = [[2, 0.95], [1, 0.80], [3, 0.40]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = compressor.compress_documents(sample_documents, "test query")

    assert len(result) == 3
    # Document 2 (index 2 -> sample_documents[1]) ranked highest
    assert result[0].page_content == sample_documents[1].page_content
    assert result[0].metadata["relevance_score"] == 0.95
    # Document 1 (index 1 -> sample_documents[0]) ranked second
    assert result[1].page_content == sample_documents[0].page_content
    assert result[1].metadata["relevance_score"] == 0.80
    # Document 3 (index 3 -> sample_documents[2]) ranked third
    assert result[2].page_content == sample_documents[2].page_content
    assert result[2].metadata["relevance_score"] == 0.40

    assert mock_engine._run_as_sync.called


@pytest.mark.asyncio
async def test_compress_documents_top_n_filtering(mock_engine, sample_documents):
    """Test that top_n restricts the number of returned documents."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[2, 0.98], [1, 0.85], [3, 0.60]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine, top_n=1)
    result = await compressor.acompress_documents(sample_documents, "query")
    assert len(result) == 1
    assert result[0].page_content == sample_documents[1].page_content
    assert result[0].metadata["relevance_score"] == 0.98


@pytest.mark.asyncio
async def test_compress_documents_immutability_guarantee(mock_engine):
    """Verify input documents are not mutated in place (SEC-01)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[1, 0.95]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    input_doc = Document(page_content="Test content", metadata={"source": "test"})
    docs = [input_doc]

    output = await compressor.acompress_documents(docs, "query")

    assert len(output) == 1
    assert output[0] is not input_doc
    assert "relevance_score" not in input_doc.metadata
    assert input_doc.metadata == {"source": "test"}
    assert output[0].metadata == {"source": "test", "relevance_score": 0.95}


@pytest.mark.asyncio
async def test_compress_documents_one_based_index_bounds(mock_engine, sample_documents):
    """Verify that 1-based PostgreSQL ordinality maps accurately (SEC-02)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    # 1 maps to documents[0], 3 maps to documents[2]
    exec_result.fetchall.return_value = [[1, 0.90], [3, 0.75]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "search query")

    assert len(result) == 2
    assert result[0].page_content == sample_documents[0].page_content
    assert result[0].metadata["relevance_score"] == 0.90
    assert result[1].page_content == sample_documents[2].page_content
    assert result[1].metadata["relevance_score"] == 0.75


@pytest.mark.asyncio
async def test_compress_documents_zero_and_negative_index_rejection(
    mock_engine, sample_documents
):
    """Verify that non-positive indices (0, -1) are rejected and skipped (SEC-02)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[0, 0.99], [-1, 0.95]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "search query")

    # Neither row should be mapped to sample_documents[-1]
    assert len(result) == 0


@pytest.mark.asyncio
async def test_compress_documents_out_of_bounds_index_rejection(
    mock_engine, sample_documents
):
    """Verify that out-of-bounds indices are safely skipped (SEC-02)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[999, 0.90], [2, 0.85]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")

    assert len(result) == 1
    assert result[0].page_content == sample_documents[1].page_content
    assert result[0].metadata["relevance_score"] == 0.85


@pytest.mark.asyncio
async def test_compress_documents_null_score_skipped(mock_engine, sample_documents):
    """Verify that NULL database scores are skipped without index-to-score fallback (SEC-03)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[2, None], [1, 0.85]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")

    assert len(result) == 1
    # Document 1 returned with score 0.85; document 2 with NULL score was skipped
    assert result[0].page_content == sample_documents[0].page_content
    assert result[0].metadata["relevance_score"] == 0.85


@pytest.mark.asyncio
async def test_compress_documents_unaligned_row_count_safety(mock_engine):
    """Verify that row count exceeding document count does not cause IndexError (SEC-04)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [
        ["invalid_1", 0.9],
        ["invalid_2", 0.8],
        [999, 0.7],
        [0, 0.6],
        [-5, 0.5],
    ]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    docs = [Document(page_content="Single document")]

    result = await compressor.acompress_documents(docs, "query")
    assert result == []


@pytest.mark.asyncio
async def test_compress_documents_single_column_skipped_safely(
    mock_engine, sample_documents
):
    """Verify single-column rows without index are skipped, avoiding ranking inversion (SEC-05)."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [[0.95], [0.10]]
    conn_mock.execute.return_value = exec_result

    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")

    # Single column rows without document index cannot be safely mapped
    assert result == []


@pytest.mark.asyncio
async def test_compress_documents_validation_empty_whitespace_query(
    mock_engine, sample_documents
):
    """Verify client-side validation rejects empty or whitespace queries (SEC-06)."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)

    with pytest.raises(ValueError, match="Query string cannot be empty or whitespace."):
        await compressor.acompress_documents(sample_documents, "")

    with pytest.raises(ValueError, match="Query string cannot be empty or whitespace."):
        await compressor.acompress_documents(sample_documents, "   \n\t  ")

    assert not mock_engine._pool.connect.called


@pytest.mark.asyncio
async def test_compress_documents_validation_non_positive_top_n(
    mock_engine, sample_documents
):
    """Verify client-side validation rejects top_n <= 0 (SEC-06)."""
    compressor_zero = AlloyDBDocumentCompressor(engine=mock_engine, top_n=0)
    with pytest.raises(
        ValueError, match="top_n must be a positive integer greater than 0."
    ):
        await compressor_zero.acompress_documents(sample_documents, "query")

    compressor_neg = AlloyDBDocumentCompressor(engine=mock_engine, top_n=-1)
    with pytest.raises(
        ValueError, match="top_n must be a positive integer greater than 0."
    ):
        await compressor_neg.acompress_documents(sample_documents, "query")

    assert not mock_engine._pool.connect.called


@pytest.mark.asyncio
async def test_compress_documents_cross_loop_event_loop_safety():
    """Verify that query execution routes via engine._run_as_async for cross-loop safety (SEC-07)."""
    engine_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=engine_loop.run_forever, daemon=True)
    thread.start()

    class CrossLoopEngine(AlloyDBEngine):

        def __init__(self):
            pass

    engine = CrossLoopEngine()
    pool_mock = MagicMock()

    async def connect_sim():
        current_loop = asyncio.get_running_loop()
        if current_loop != engine_loop:
            raise RuntimeError("Task got Future attached to a different loop")
        conn = AsyncMock()
        res = MagicMock()
        res.fetchall.return_value = [[1, 0.95]]
        conn.execute.return_value = res
        return conn

    class SimContext:

        async def __aenter__(self):
            return await connect_sim()

        async def __aexit__(self, *args):
            pass

    pool_mock.connect.side_effect = lambda: SimContext()
    engine._pool = pool_mock

    async def cross_loop_run_as_async(coro):
        future = asyncio.run_coroutine_threadsafe(coro, engine_loop)
        return await asyncio.wrap_future(future)

    engine._run_as_async = MagicMock(side_effect=cross_loop_run_as_async)

    try:
        compressor = AlloyDBDocumentCompressor(engine=engine)
        docs = [Document(page_content="doc1")]
        result = await compressor.acompress_documents(docs, "query")
        assert len(result) == 1
        assert result[0].metadata["relevance_score"] == 0.95
        assert engine._run_as_async.called
    finally:
        engine_loop.call_soon_threadsafe(engine_loop.stop)
        thread.join(timeout=2.0)
        engine_loop.close()


@pytest.mark.asyncio
async def test_compress_documents_infinity_index_overflow_handling(
    mock_engine, sample_documents
):
    """Verify that float('inf') or float('-inf') index rows are safely skipped without OverflowError (NEW-641-01)."""
    conn = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        [float("inf"), 0.99],
        [float("-inf"), 0.88],
        [1, 0.95],
    ]
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")
    assert len(result) == 1
    assert result[0].page_content == sample_documents[0].page_content
    assert result[0].metadata["relevance_score"] == 0.95


@pytest.mark.asyncio
async def test_compress_documents_array_fallback_non_numeric_score(
    mock_engine, sample_documents
):
    """Verify that non-numeric scores in array fallback are safely skipped without ValueError (NEW-641-02)."""
    conn = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        [["not_a_number", 0.85, "nan_val"]]
    ]
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")
    assert len(result) == 1
    assert result[0].page_content == sample_documents[1].page_content
    assert result[0].metadata["relevance_score"] == 0.85


@pytest.mark.asyncio
async def test_compress_documents_fractional_float_index_rejected(
    mock_engine, sample_documents
):
    """Verify that fractional float indices (e.g. 1.5) are safely skipped rather than truncated (NEW-641-03)."""
    conn = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        [1.5, 0.99],
        [2.7, 0.88],
        [1.0, 0.95],  # 1.0 is an exact integer float, should be accepted
    ]
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    result = await compressor.acompress_documents(sample_documents, "query")
    assert len(result) == 1
    assert result[0].page_content == sample_documents[0].page_content
    assert result[0].metadata["relevance_score"] == 0.95


def test_compress_documents_sync_deadlock_on_engine_loop():
    """Verify that calling compress_documents on the engine loop raises RuntimeError (NEW-641-04)."""
    loop = asyncio.new_event_loop()

    class DeadlockEngine(AlloyDBEngine):

        def __init__(self, loop):
            self._loop = loop

    engine = DeadlockEngine(loop)
    compressor = AlloyDBDocumentCompressor(engine=engine)

    def run_on_loop():
        with pytest.raises(
            RuntimeError, match="Cannot call synchronous 'compress_documents'"
        ):
            compressor.compress_documents([Document(page_content="text")], "query")

    async def runner():
        run_on_loop()

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()
