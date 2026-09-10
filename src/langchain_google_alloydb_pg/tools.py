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
import inspect
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
    """Exception raised when an AlloyDB AI tool encounters an execution error."""

    pass


class SentimentInput(BaseModel):
    """Input for AlloyDBSentimentTool."""

    content: str = Field(
        description="The text content to analyze sentiment for.",
        min_length=1,
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional registered model ID in AlloyDB to use for sentiment analysis.",
    )

    @field_validator("content")
    @classmethod
    def check_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("model_id")
    @classmethod
    def clean_model_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Model ID cannot be whitespace only.")
        return v.strip() if v else None


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
    model_id: Optional[str] = Field(
        default=None,
        description="Optional registered model ID in AlloyDB to use for summarization.",
    )

    @field_validator("content")
    @classmethod
    def check_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("additional_instructions")
    @classmethod
    def clean_additional_instructions(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Additional instructions cannot be whitespace only.")
        return v.strip() if v else None

    @field_validator("model_id")
    @classmethod
    def clean_model_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Model ID cannot be whitespace only.")
        return v.strip() if v else None


class AlloyDBIfInput(BaseModel):
    """Input for AlloyDBIfTool."""

    condition: str = Field(
        description=(
            "The semantic condition to evaluate (e.g. 'Is the text positive?')"
        ),
        min_length=1,
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional registered model ID in AlloyDB to use for evaluation.",
    )

    @field_validator("condition")
    @classmethod
    def check_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Condition cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("model_id")
    @classmethod
    def clean_model_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Model ID cannot be whitespace only.")
        return v.strip() if v else None


def _check_engine_deadlock(engine: AlloyDBEngine) -> None:
    try:
        curr_loop = asyncio.get_running_loop()
    except RuntimeError:
        curr_loop = None

    if (
        curr_loop is not None
        and getattr(engine, "_loop", None) is not None
        and curr_loop == engine._loop
    ):
        raise RuntimeError(
            "Cannot call synchronous '_run' from the engine's background event loop "
            "as it causes a thread deadlock. Use '_arun' or 'ainvoke' instead."
        )


def _invoke_sync_callbacks_start(
    run_manager: Optional[CallbackManagerForToolRun], input_data: Any
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_start"):
        try:
            run_manager.on_tool_start(input_data)
        except Exception:
            pass


def _invoke_sync_callbacks_end(
    run_manager: Optional[CallbackManagerForToolRun], output: Any
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_end"):
        try:
            run_manager.on_tool_end(output)
        except Exception:
            pass


def _invoke_sync_callbacks_error(
    run_manager: Optional[CallbackManagerForToolRun], error: BaseException
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_error"):
        try:
            run_manager.on_tool_error(error)
        except Exception:
            pass


async def _invoke_async_callbacks_start(
    run_manager: Optional[AsyncCallbackManagerForToolRun], input_data: Any
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_start"):
        try:
            res = run_manager.on_tool_start(input_data)
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass


async def _invoke_async_callbacks_end(
    run_manager: Optional[AsyncCallbackManagerForToolRun], output: Any
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_end"):
        try:
            res = run_manager.on_tool_end(output)
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass


async def _invoke_async_callbacks_error(
    run_manager: Optional[AsyncCallbackManagerForToolRun],
    error: BaseException,
) -> None:
    if run_manager is not None and hasattr(run_manager, "on_tool_error"):
        try:
            res = run_manager.on_tool_error(error)
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass


class AlloyDBSentimentTool(BaseTool):
    """Tool for analyzing sentiment of text using AlloyDB AI functions.

    Note: Requires AlloyDB running PostgreSQL 17 or higher with google_ml_integration.
    """

    name: str = "alloydb_sentiment_tool"
    description: str = (
        "Analyze the sentiment of a given text. Useful for determining if text is"
        " positive, negative, or neutral."
    )
    args_schema: Type[BaseModel] = SentimentInput
    engine: AlloyDBEngine
    model_id: Optional[str] = None
    handle_tool_error: bool = True

    def _run(
        self,
        content: str,
        model_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool synchronously."""
        if isinstance(
            model_id,
            (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
        ) or (
            model_id is not None
            and not isinstance(model_id, str)
            and (hasattr(model_id, "on_tool_start") or hasattr(model_id, "on_tool_end"))
        ):
            run_manager = model_id  # type: ignore
            model_id = None

        _check_engine_deadlock(self.engine)
        _invoke_sync_callbacks_start(run_manager, {"content": content})
        try:
            result = self.engine._run_as_sync(
                self.__arun(content, model_id=model_id, run_manager=None)
            )
            _invoke_sync_callbacks_end(run_manager, result)
            return result
        except Exception as e:
            _invoke_sync_callbacks_error(run_manager, e)
            raise

    async def _arun(
        self,
        content: str,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        if isinstance(
            model_id,
            (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
        ) or (
            model_id is not None
            and not isinstance(model_id, str)
            and (hasattr(model_id, "on_tool_start") or hasattr(model_id, "on_tool_end"))
        ):
            run_manager = model_id  # type: ignore
            model_id = None

        await _invoke_async_callbacks_start(run_manager, {"content": content})
        try:
            result = await self.engine._run_as_async(
                self.__arun(content, model_id=model_id, run_manager=None)
            )
            await _invoke_async_callbacks_end(run_manager, result)
            return result
        except Exception as e:
            await _invoke_async_callbacks_error(run_manager, e)
            raise

    async def __arun(
        self,
        content: str,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        effective_model_id = model_id or self.model_id
        if effective_model_id:
            query = "SELECT google_ml.analyze_sentiment(:content, :model_id)"
            params = {"content": content, "model_id": effective_model_id}
        else:
            query = "SELECT google_ml.analyze_sentiment(:content)"
            params = {"content": content}

        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), params)
            val = result.scalar()
            if val is None:
                raise AlloyDBToolError(
                    "AlloyDB AI sentiment analysis returned NULL. "
                    "Ensure the input is valid and the model is accessible."
                )
            return str(val)


class AlloyDBSummaryTool(BaseTool):
    """Tool for summarizing text using AlloyDB AI functions.

    Note: Requires AlloyDB running PostgreSQL 17 or higher with google_ml_integration.
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
    handle_tool_error: bool = True

    def _run(
        self,
        content: str,
        additional_instructions: Optional[str] = None,
        model_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool synchronously."""
        for pos_arg_name in ("model_id", "additional_instructions"):
            val = locals()[pos_arg_name]
            if isinstance(
                val,
                (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
            ) or (
                val is not None
                and not isinstance(val, str)
                and (hasattr(val, "on_tool_start") or hasattr(val, "on_tool_end"))
            ):
                run_manager = val  # type: ignore
                if pos_arg_name == "model_id":
                    model_id = None
                else:
                    additional_instructions = None

        _check_engine_deadlock(self.engine)
        _invoke_sync_callbacks_start(run_manager, {"content": content})
        try:
            result = self.engine._run_as_sync(
                self.__arun(
                    content,
                    additional_instructions=additional_instructions,
                    model_id=model_id,
                    run_manager=None,
                )
            )
            _invoke_sync_callbacks_end(run_manager, result)
            return result
        except Exception as e:
            _invoke_sync_callbacks_error(run_manager, e)
            raise

    async def _arun(
        self,
        content: str,
        additional_instructions: Optional[str] = None,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        for pos_arg_name in ("model_id", "additional_instructions"):
            val = locals()[pos_arg_name]
            if isinstance(
                val,
                (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
            ) or (
                val is not None
                and not isinstance(val, str)
                and (hasattr(val, "on_tool_start") or hasattr(val, "on_tool_end"))
            ):
                run_manager = val  # type: ignore
                if pos_arg_name == "model_id":
                    model_id = None
                else:
                    additional_instructions = None

        await _invoke_async_callbacks_start(run_manager, {"content": content})
        try:
            result = await self.engine._run_as_async(
                self.__arun(
                    content,
                    additional_instructions=additional_instructions,
                    model_id=model_id,
                    run_manager=None,
                )
            )
            await _invoke_async_callbacks_end(run_manager, result)
            return result
        except Exception as e:
            await _invoke_async_callbacks_error(run_manager, e)
            raise

    async def __arun(
        self,
        content: str,
        additional_instructions: Optional[str] = None,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        effective_model_id = model_id or self.model_id
        effective_instructions = additional_instructions or self.additional_instructions

        if effective_model_id and effective_instructions:
            query = (
                "SELECT google_ml.summarize(:content, :additional_instructions,"
                " :model_id)"
            )
            params = {
                "content": content,
                "additional_instructions": effective_instructions,
                "model_id": effective_model_id,
            }
        elif effective_instructions:
            query = "SELECT google_ml.summarize(:content, :additional_instructions)"
            params = {
                "content": content,
                "additional_instructions": effective_instructions,
            }
        elif effective_model_id:
            query = "SELECT google_ml.summarize(:content, NULL, :model_id)"
            params = {
                "content": content,
                "model_id": effective_model_id,
            }
        else:
            query = "SELECT google_ml.summarize(:content)"
            params = {"content": content}

        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), params)
            val = result.scalar()
            if val is None:
                raise AlloyDBToolError(
                    "AlloyDB AI text summarization returned NULL. "
                    "Ensure the input is valid and the model is accessible."
                )
            return str(val)


class AlloyDBIfTool(BaseTool):
    """Tool that evaluates a semantic condition using AlloyDB AI google_ml.if function.

    Note: Supported on AlloyDB running PostgreSQL 14 or higher with google_ml_integration.
    """

    name: str = "alloydb_if_tool"
    description: str = (
        "A tool that uses AlloyDB AI to evaluate a semantic condition and"
        " returns True or False. Useful for semantic routing, classification,"
        " or filtering."
    )
    args_schema: Type[BaseModel] = AlloyDBIfInput
    engine: AlloyDBEngine
    model_id: Optional[str] = None
    handle_tool_error: bool = True

    def _run(
        self,
        condition: str,
        model_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> bool:
        """Evaluate the condition synchronously."""
        if isinstance(
            model_id,
            (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
        ) or (
            model_id is not None
            and not isinstance(model_id, str)
            and (hasattr(model_id, "on_tool_start") or hasattr(model_id, "on_tool_end"))
        ):
            run_manager = model_id  # type: ignore
            model_id = None

        _check_engine_deadlock(self.engine)
        _invoke_sync_callbacks_start(run_manager, {"condition": condition})
        try:
            result = self.engine._run_as_sync(
                self.__arun(condition, model_id=model_id, run_manager=None)
            )
            _invoke_sync_callbacks_end(run_manager, result)
            return bool(result)
        except Exception as e:
            _invoke_sync_callbacks_error(run_manager, e)
            raise

    async def _arun(
        self,
        condition: str,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> bool:
        """Evaluate the condition asynchronously."""
        if isinstance(
            model_id,
            (CallbackManagerForToolRun, AsyncCallbackManagerForToolRun),
        ) or (
            model_id is not None
            and not isinstance(model_id, str)
            and (hasattr(model_id, "on_tool_start") or hasattr(model_id, "on_tool_end"))
        ):
            run_manager = model_id  # type: ignore
            model_id = None

        await _invoke_async_callbacks_start(run_manager, {"condition": condition})
        try:
            result = await self.engine._run_as_async(
                self.__arun(condition, model_id=model_id, run_manager=None)
            )
            await _invoke_async_callbacks_end(run_manager, result)
            return bool(result)
        except Exception as e:
            await _invoke_async_callbacks_error(run_manager, e)
            raise

    async def __arun(
        self,
        condition: str,
        model_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> bool:
        effective_model_id = model_id or self.model_id
        if effective_model_id:
            query = "SELECT google_ml.if(:condition, :model_id)"
            params = {"condition": condition, "model_id": effective_model_id}
        else:
            query = "SELECT google_ml.if(:condition)"
            params = {"condition": condition}

        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), params)
            val = result.scalar()
            if val is None:
                raise AlloyDBToolError(
                    "AlloyDB AI google_ml.if returned NULL (evaluation ambiguous or failed)."
                )
            return bool(val)
