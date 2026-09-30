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
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .engine import AlloyDBEngine


class AlloyDBToolError(ToolException, ValueError):
    """Raised when an AlloyDB AI function returns NULL.

    That happens, for example, when the model's reply isn't one of the
    expected answers. ``invoke`` / ``ainvoke`` raise this error like any
    other. Because it is a ``ToolException``, a tool created with
    ``handle_tool_error=True`` returns its message to the agent instead.
    Database errors (``DBAPIError``), such as an unknown model, always raise.
    """


def _check_text(value: str, name: str) -> str:
    """Reject empty or whitespace-only text. The value is used as given."""
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _check_model_id(value: Optional[str]) -> Optional[str]:
    """Reject a blank model ID. ``None`` selects the database's default model."""
    return None if value is None else _check_text(value, "model_id")


def _check_engine_loop(engine: AlloyDBEngine) -> None:
    """Raise if a synchronous call would block the engine's own event loop."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if running_loop is engine._loop:
        raise RuntimeError(
            "Cannot call a synchronous tool method from the engine's event loop "
            "because it would deadlock. Use 'ainvoke' instead."
        )


async def _afetch_scalar(
    engine: AlloyDBEngine, query: str, params: dict[str, Any], null_message: str
) -> Any:
    """Run ``query`` and return its single value, raising if it is NULL."""
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query), params)
        value = result.scalar()
    if value is None:
        raise AlloyDBToolError(null_message)
    return value


class SentimentInput(BaseModel):
    """Input for AlloyDBSentimentTool."""

    content: str = Field(
        description="The text content to analyze sentiment for.",
        min_length=1,
    )

    @field_validator("content")
    @classmethod
    def check_content(cls, v: str) -> str:
        return _check_text(v, "content")


class AlloyDBSentimentTool(BaseTool):
    """Analyzes the sentiment of text with ``google_ml.analyze_sentiment``.

    Requires AlloyDB running PostgreSQL 17 or higher with
    ``google_ml_integration``. Like any LangChain tool, call it with
    ``invoke`` / ``ainvoke`` or give it to an agent:

    .. code-block:: python

        tool = AlloyDBSentimentTool(engine=engine)
        tool.invoke({"content": "I love this product!"})  # "positive"

    ``model_id`` is set on the tool, not by the agent. If it is ``None``, the
    database uses the model named by its
    ``google_ml_integration.default_llm_model`` flag. If the function returns
    NULL, the tool raises :class:`AlloyDBToolError`.
    """

    name: str = "alloydb_sentiment_tool"
    description: str = (
        "Analyze the sentiment of a given text. Useful for determining if text is"
        " positive, negative, or neutral."
    )
    args_schema: Type[BaseModel] = SentimentInput
    engine: AlloyDBEngine
    model_id: Optional[str] = None

    @field_validator("model_id")
    @classmethod
    def check_model_id(cls, v: Optional[str]) -> Optional[str]:
        return _check_model_id(v)

    def _run(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Analyze sentiment synchronously. Called by ``invoke`` and ``run``."""
        _check_engine_loop(self.engine)
        return self.engine._run_as_sync(self.__aanalyze_sentiment(content))

    async def _arun(
        self,
        content: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Analyze sentiment asynchronously. Called by ``ainvoke`` and ``arun``."""
        return await self.engine._run_as_async(self.__aanalyze_sentiment(content))

    async def __aanalyze_sentiment(self, content: str) -> str:
        if self.model_id:
            query = "SELECT google_ml.analyze_sentiment(:content, :model_id)"
            params = {"content": content, "model_id": self.model_id}
        else:
            query = "SELECT google_ml.analyze_sentiment(:content)"
            params = {"content": content}
        value = await _afetch_scalar(
            self.engine,
            query,
            params,
            "AlloyDB AI sentiment analysis returned NULL: the model's reply "
            "wasn't positive, negative or neutral.",
        )
        return str(value)


class SummaryInput(BaseModel):
    """Input for AlloyDBSummaryTool."""

    content: str = Field(
        description="The text content to summarize.",
        min_length=1,
    )
    additional_instructions: Optional[str] = Field(
        default=None,
        description="Optional additional instructions for summarization style or format.",
    )

    @field_validator("content")
    @classmethod
    def check_content(cls, v: str) -> str:
        return _check_text(v, "content")

    @field_validator("additional_instructions")
    @classmethod
    def check_additional_instructions(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else _check_text(v, "additional_instructions")


class AlloyDBSummaryTool(BaseTool):
    """Summarizes text with ``google_ml.summarize``.

    Requires AlloyDB running PostgreSQL 17 or higher with
    ``google_ml_integration``. Like any LangChain tool, call it with
    ``invoke`` / ``ainvoke`` or give it to an agent:

    .. code-block:: python

        tool = AlloyDBSummaryTool(engine=engine)
        tool.invoke(
            {"content": article, "additional_instructions": "In 3 bullet points"}
        )

    ``model_id`` works as in :class:`AlloyDBSentimentTool`: it is set on the
    tool, and ``None`` uses the database's default model.
    ``additional_instructions`` can be set on the tool or passed per call; a
    per-call value replaces the tool's.
    """

    name: str = "alloydb_summary_tool"
    description: str = (
        "Summarize the given text. Useful for condensing long articles or"
        " descriptions into shorter summaries."
    )
    args_schema: Type[BaseModel] = SummaryInput
    engine: AlloyDBEngine
    model_id: Optional[str] = None
    additional_instructions: Optional[str] = None

    @field_validator("model_id")
    @classmethod
    def check_model_id(cls, v: Optional[str]) -> Optional[str]:
        return _check_model_id(v)

    @field_validator("additional_instructions")
    @classmethod
    def check_additional_instructions(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else _check_text(v, "additional_instructions")

    def _run(
        self,
        content: str,
        additional_instructions: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Summarize synchronously. Called by ``invoke`` and ``run``."""
        _check_engine_loop(self.engine)
        return self.engine._run_as_sync(
            self.__asummarize(content, additional_instructions)
        )

    async def _arun(
        self,
        content: str,
        additional_instructions: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Summarize asynchronously. Called by ``ainvoke`` and ``arun``."""
        return await self.engine._run_as_async(
            self.__asummarize(content, additional_instructions)
        )

    async def __asummarize(
        self, content: str, additional_instructions: Optional[str]
    ) -> str:
        model_id = self.model_id
        instructions = additional_instructions or self.additional_instructions
        params = {"content": content}
        if instructions and model_id:
            query = "SELECT google_ml.summarize(:content, :instructions, :model_id)"
            params.update(instructions=instructions, model_id=model_id)
        elif instructions:
            query = "SELECT google_ml.summarize(:content, :instructions)"
            params.update(instructions=instructions)
        elif model_id:
            query = "SELECT google_ml.summarize(:content, NULL, :model_id)"
            params.update(model_id=model_id)
        else:
            query = "SELECT google_ml.summarize(:content)"
        value = await _afetch_scalar(
            self.engine,
            query,
            params,
            "AlloyDB AI text summarization returned NULL: the model's reply "
            "had no text.",
        )
        return str(value)
