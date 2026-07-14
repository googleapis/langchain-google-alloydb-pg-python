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
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = await tool._arun("I love this!")
        assert result == "mocked_result"
        
    def test_sentiment_tool_run(self, mock_engine):
        tool = AlloyDBSentimentTool(engine=mock_engine)
        result = tool._run("I love this!")
        assert result == "mocked_result"

    async def test_summary_tool_arun(self, mock_engine):
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = await tool._arun("A very long article goes here.")
        assert result == "mocked_result"
        
    def test_summary_tool_run(self, mock_engine):
        tool = AlloyDBSummaryTool(engine=mock_engine)
        result = tool._run("A very long article goes here.")
        assert result == "mocked_result"
