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

from typing import List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from .engine import AlloyDBEngine


class NL2SQLInput(BaseModel):
    query: str = Field(description="The natural language query to translate to SQL.")


class AlloyDBNL2SQLTool(BaseTool):
    """Tool for translating natural language to SQL using AlloyDB's native NL2SQL."""

    name: str = "alloydb_nl2sql_tool"
    description: str = (
        "Translate a natural language question into a SQL query using AlloyDB AI. "
        "Pass the user's natural language question as the query."
    )
    args_schema: Type[BaseModel] = NL2SQLInput
    engine: AlloyDBEngine

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool synchronously."""
        return self.engine._run_as_sync(self._arun(query, run_manager))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        # Using ai.generate_sql based on general AlloyDB AI syntax patterns
        # or similar functions to generate SQL natively.
        sql_query = "SELECT google_ml.generate_sql(:query)"
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(sql_query), {"query": query})
            return str(result.scalar())


class AlloyDBToolkit(BaseToolkit):
    """Toolkit for interacting with AlloyDB databases using native AI features."""

    engine: AlloyDBEngine = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        nl2sql_tool = AlloyDBNL2SQLTool(engine=self.engine)
        return [nl2sql_tool]
