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

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_google_alloydb_pg.tools import AlloyDBSentimentTool, AlloyDBSummaryTool

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    # Mock synchronous run mapping
    engine._run_as_sync = MagicMock(side_effect=lambda coro: "mocked_result")
    
    # Mock pool and connection for async runs
    pool_mock = MagicMock()
    conn_mock = AsyncMock()
    conn_mock.execute.return_value = MagicMock(scalar=MagicMock(return_value="mocked_result"))
    pool_mock.connect.return_value.__aenter__.return_value = conn_mock
    engine._pool = pool_mock
    
    return engine

@pytest.mark.asyncio
class TestAlloyDBTools:
    
    async def test_sentiment_tool_arun(self, mock_engine):
        """Test that AlloyDBSentimentTool._arun executes the sentiment analysis SQL asynchronously."""
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = await tool._arun("I love this!")
        assert result == "mocked_result"
        # Verify the exact SQL query
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        executed_query = conn_mock.execute.call_args[0][0].text
        assert "SELECT google_ml.sentiment_analysis" in executed_query

    def test_sentiment_tool_run(self, mock_engine):
        """Test that AlloyDBSentimentTool._run executes the sentiment analysis SQL synchronously."""
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = tool._run("I love this!")
        assert result == "mocked_result"

    async def test_summary_tool_arun(self, mock_engine):
        """Test that AlloyDBSummaryTool._arun executes the text summarization SQL asynchronously."""
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = await tool._arun("A very long article goes here.")
        assert result == "mocked_result"
        # Verify the exact SQL query
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        executed_query = conn_mock.execute.call_args[0][0].text
        assert "SELECT google_ml.summarize_content" in executed_query
        
    def test_summary_tool_run(self, mock_engine):
        """Test that AlloyDBSummaryTool._run executes the text summarization SQL synchronously."""
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = tool._run("A very long article goes here.")
        assert result == "mocked_result"
