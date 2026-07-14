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
        """Test that AlloyDBNL2SQLTool._arun executes the NL2SQL generation asynchronously."""
        tool = AlloyDBNL2SQLTool(engine=mock_engine)
        result = await tool._arun("Show me all users")
        assert result == "mocked_sql_query"
        # Verify the exact SQL query
        conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
        executed_query = conn_mock.execute.call_args[0][0].text
        assert "SELECT google_ml.generate_sql" in executed_query

    def test_nl2sql_tool_run(self, mock_engine):
        """Test that AlloyDBNL2SQLTool._run executes the NL2SQL generation synchronously."""
        tool = AlloyDBNL2SQLTool(engine=mock_engine)
        result = tool._run("Show me all users")
        assert result == "mocked_sql_query"

    def test_toolkit_get_tools(self, mock_engine):
        """Test that AlloyDBToolkit properly exposes its internal list of AI tools."""
        toolkit = AlloyDBToolkit(engine=mock_engine)
        tools = toolkit.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], AlloyDBNL2SQLTool)
