import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_google_alloydb_pg.toolkit import AlloyDBToolkit, AlloyDBNL2SQLTool

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    # Mock synchronous run mapping
    engine._run_as_sync = MagicMock(side_effect=lambda coro: "mocked_sql_query")
    
    # Mock pool and connection for async runs
    pool_mock = MagicMock()
    conn_mock = AsyncMock()
    conn_mock.execute.return_value = MagicMock(scalar=MagicMock(return_value="mocked_sql_query"))
    pool_mock.connect.return_value.__aenter__.return_value = conn_mock
    engine._pool = pool_mock
    
    return engine

@pytest.mark.asyncio
class TestAlloyDBToolkit:
    
    async def test_nl2sql_tool_arun(self, mock_engine):
        tool = AlloyDBNL2SQLTool(engine=mock_engine)
        result = await tool._arun("Show me all users")
        assert result == "mocked_sql_query"
        
    def test_nl2sql_tool_run(self, mock_engine):
        tool = AlloyDBNL2SQLTool(engine=mock_engine)
        result = tool._run("Show me all users")
        assert result == "mocked_sql_query"

    def test_toolkit_get_tools(self, mock_engine):
        toolkit = AlloyDBToolkit(engine=mock_engine)
        tools = toolkit.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], AlloyDBNL2SQLTool)
