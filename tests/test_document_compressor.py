# Copyright 2025 Google LLC
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

import pytest
from langchain_core.documents import Document

from langchain_google_alloydb_pg import AlloyDBDocumentCompressor

@pytest.fixture
def mock_documents():
    return [
        Document(page_content="The quick brown fox."),
        Document(page_content="Jumps over the lazy dog."),
    ]

@pytest.mark.asyncio
async def test_compressor_arun(mock_engine, mock_documents):
    """Test that AlloyDBDocumentCompressor executes the google_ml.rank function asynchronously."""
    # We must configure our mock to return a row with a score
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.fetchall.return_value = [[0.9], [0.1]]

    compressor = AlloyDBDocumentCompressor(engine=mock_engine, model_id="semantic-ranker-512@latest", top_n=2)
    result = await compressor.acompress_documents(mock_documents, "query about a fox")
    
    assert len(result) == 2
    assert result[0].page_content == "The quick brown fox."
    assert result[0].metadata["relevance_score"] == 0.9
    
    # Verify the exact SQL query
    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT * FROM google_ml.rank" in executed_query

def test_compressor_run(mock_engine, mock_documents):
    """Test that AlloyDBDocumentCompressor executes synchronously."""
    compressor = AlloyDBDocumentCompressor(engine=mock_engine)
    # Mock engine._run_as_sync directly for the sync wrapper
    import unittest.mock
    
    # Pre-configure the documents with mock relevance scores
    mock_docs_ranked = [
        Document(page_content="The quick brown fox.", metadata={"relevance_score": 0.9}),
    ]
    with unittest.mock.patch.object(mock_engine, "_run_as_sync", return_value=mock_docs_ranked):
        result = compressor.compress_documents(mock_documents, "query about a fox")
        assert len(result) == 1
        assert result[0].metadata["relevance_score"] == 0.9
