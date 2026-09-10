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

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_google_alloydb_pg.engine import AlloyDBEngine
from langchain_google_alloydb_pg.tools import (
    AlloyDBIfInput,
    AlloyDBIfTool,
    AlloyDBSentimentTool,
    AlloyDBSummaryTool,
    SentimentInput,
    SummaryInput,
)


@pytest.fixture
def mock_engine():
    """Fixture providing a mock AlloyDBEngine for offline testing."""

    class DummyEngine(AlloyDBEngine):

        def __init__(self):
            pass

    engine = DummyEngine()

    def _sync_side_effect(coro):
        coro.close()
        return "mocked_result"

    engine._run_as_sync = MagicMock(side_effect=_sync_side_effect)

    # Mock pool and connection for async runs
    pool_mock = MagicMock()
    conn_mock = AsyncMock()
    conn_mock.execute.return_value = MagicMock(
        scalar=MagicMock(return_value="mocked_result")
    )
    pool_mock.connect.return_value.__aenter__.return_value = conn_mock
    engine._pool = pool_mock

    return engine


class TestAlloyDBTools:
    """Unit tests for AlloyDB GenAI tools."""

    def test_tool_metadata(self, mock_engine):
        """Test tool attributes, schemas, and names."""
        sentiment_tool = AlloyDBSentimentTool(engine=mock_engine)
        assert sentiment_tool.name == "alloydb_sentiment_tool"
        assert "sentiment" in sentiment_tool.description.lower()
        assert sentiment_tool.args_schema == SentimentInput

        summary_tool = AlloyDBSummaryTool(engine=mock_engine)
        assert summary_tool.name == "alloydb_summary_tool"
        assert "summarize" in summary_tool.description.lower()
        assert summary_tool.args_schema == SummaryInput

        if_tool = AlloyDBIfTool(engine=mock_engine)
        assert if_tool.name == "alloydb_if"
        assert "semantic condition" in if_tool.description.lower()
        assert if_tool.args_schema == AlloyDBIfInput

    @pytest.mark.asyncio
    async def test_sentiment_tool_arun(self, mock_engine):
        """Test that AlloyDBSentimentTool._arun executes sentiment analysis SQL asynchronously."""
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = await tool._arun("I love this!")
        assert result == "mocked_result"

        # Verify the exact SQL query and parameters
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        executed_query = conn_mock.execute.call_args[0][0].text
        executed_params = conn_mock.execute.call_args[0][1]
        assert "SELECT google_ml.analyze_sentiment" in executed_query
        assert executed_params == {"content": "I love this!"}

    def test_sentiment_tool_run(self, mock_engine):
        """Test that AlloyDBSentimentTool._run executes sentiment analysis SQL synchronously."""
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = tool._run("I love this!")
        assert result == "mocked_result"

    @pytest.mark.asyncio
    async def test_summary_tool_arun(self, mock_engine):
        """Test that AlloyDBSummaryTool._arun executes text summarization SQL asynchronously."""
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = await tool._arun("A very long article goes here.")
        assert result == "mocked_result"

        # Verify the exact SQL query and parameters
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        executed_query = conn_mock.execute.call_args[0][0].text
        executed_params = conn_mock.execute.call_args[0][1]
        assert "SELECT google_ml.summarize" in executed_query
        assert executed_params == {"content": "A very long article goes here."}

    def test_summary_tool_run(self, mock_engine):
        """Test that AlloyDBSummaryTool._run executes text summarization SQL synchronously."""
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = tool._run("A very long article goes here.")
        assert result == "mocked_result"

    @pytest.mark.asyncio
    async def test_if_tool_arun_true(self, mock_engine):
        """Test that AlloyDBIfTool._arun executes boolean condition SQL returning True."""
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        conn_mock.execute.return_value.scalar.return_value = True

        tool = AlloyDBIfTool(engine=mock_engine)
        result = await tool._arun("Is this a positive review?")
        assert result is True

        # Verify the exact SQL query and parameters
        executed_query = conn_mock.execute.call_args[0][0].text
        executed_params = conn_mock.execute.call_args[0][1]
        assert "SELECT google_ml.if" in executed_query
        assert executed_params == {"condition": "Is this a positive review?"}

    @pytest.mark.asyncio
    async def test_if_tool_arun_false(self, mock_engine):
        """Test that AlloyDBIfTool._arun executes boolean condition SQL returning False."""
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        conn_mock.execute.return_value.scalar.return_value = False

        tool = AlloyDBIfTool(engine=mock_engine)
        result = await tool._arun("Is this a negative review?")
        assert result is False

        executed_query = conn_mock.execute.call_args[0][0].text
        executed_params = conn_mock.execute.call_args[0][1]
        assert "SELECT google_ml.if" in executed_query
        assert executed_params == {"condition": "Is this a negative review?"}

    def test_if_tool_run_true(self, mock_engine):
        """Test that AlloyDBIfTool._run executes synchronously returning True."""
        tool = AlloyDBIfTool(engine=mock_engine)

        def _sync_if_true(coro):
            coro.close()
            return True

        mock_engine._run_as_sync = MagicMock(side_effect=_sync_if_true)
        result = tool._run("Is this valid?")
        assert result is True

    def test_if_tool_run_false(self, mock_engine):
        """Test that AlloyDBIfTool._run executes synchronously returning False."""
        tool = AlloyDBIfTool(engine=mock_engine)

        def _sync_if_false(coro):
            coro.close()
            return False

        mock_engine._run_as_sync = MagicMock(side_effect=_sync_if_false)
        result = tool._run("Is this valid?")
        assert result is False

    @pytest.mark.asyncio
    async def test_tool_ainvoke_and_invoke(self, mock_engine):
        """Test that standard LangChain invoke and ainvoke entrypoints work with schemas."""
        # Test Sentiment Tool
        sentiment_tool = AlloyDBSentimentTool(engine=mock_engine)
        async_res = await sentiment_tool.ainvoke({"content": "Great product!"})
        assert async_res == "mocked_result"
        sync_res = sentiment_tool.invoke({"content": "Great product!"})
        assert sync_res == "mocked_result"

        # Test Summary Tool
        summary_tool = AlloyDBSummaryTool(engine=mock_engine)
        async_sum = await summary_tool.ainvoke({"content": "Long article."})
        assert async_sum == "mocked_result"
        sync_sum = summary_tool.invoke({"content": "Long article."})
        assert sync_sum == "mocked_result"

        # Test If Tool
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        conn_mock.execute.return_value.scalar.return_value = True

        def _sync_if_true(coro):
            coro.close()
            return True

        mock_engine._run_as_sync = MagicMock(side_effect=_sync_if_true)

        if_tool = AlloyDBIfTool(engine=mock_engine)
        async_if = await if_tool.ainvoke({"condition": "Is it valid?"})
        assert async_if is True
        sync_if = if_tool.invoke({"condition": "Is it valid?"})
        assert sync_if is True
