# Copyright 2024 Google LLC
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

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import text

from .engine import AlloyDBEngine


class SentimentInput(BaseModel):
    content: str = Field(description="The text content to analyze sentiment for.")


class AlloyDBSentimentTool(BaseTool):
    """Tool for analyzing sentiment of text using AlloyDB AI functions."""

    name: str = "alloydb_sentiment_tool"
    description: str = "Analyze the sentiment of a given text. Useful for determining if text is positive, negative, or neutral."
    args_schema: Type[BaseModel] = SentimentInput
    engine: AlloyDBEngine

    def _run(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool synchronously."""
        return self.engine._run_as_sync(self._arun(content, run_manager))

    async def _arun(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        query = "SELECT google_ml.analyze_sentiment(:content)"
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), {"content": content})
            return str(result.scalar())


class SummaryInput(BaseModel):
    content: str = Field(description="The text content to summarize.")


class AlloyDBSummaryTool(BaseTool):
    """Tool for summarizing text using AlloyDB AI functions."""

    name: str = "alloydb_summary_tool"
    description: str = "Summarize the given text. Useful for condensing long articles or descriptions into shorter summaries."
    args_schema: Type[BaseModel] = SummaryInput
    engine: AlloyDBEngine

    def _run(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool synchronously."""
        return self.engine._run_as_sync(self._arun(content, run_manager))

    async def _arun(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        query = "SELECT google_ml.summarize(:content)"
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), {"content": content})
            return str(result.scalar())


class AlloyDBIfInput(BaseModel):
    """Input for AlloyDBIfTool."""
    condition: str = Field(description="The semantic condition to evaluate (e.g. 'Is the text positive?')")


class AlloyDBIfTool(BaseTool):
    """Tool that evaluates a semantic condition using AlloyDB AI google_ml.if function."""

    name: str = "alloydb_if"
    description: str = (
        "A tool that uses AlloyDB AI to evaluate a semantic condition and returns True or False. "
        "Useful for semantic routing, classification, or filtering."
    )
    args_schema: Type[BaseModel] = AlloyDBIfInput
    engine: AlloyDBEngine

    def _run(
        self,
        condition: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Evaluate the condition synchronously."""
        return self.engine._run_as_sync(self._arun(condition, run_manager))

    async def _arun(
        self,
        condition: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Evaluate the condition asynchronously."""
        query = "SELECT google_ml.if(:condition)"
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), {"condition": condition})
            return bool(result.scalar())
