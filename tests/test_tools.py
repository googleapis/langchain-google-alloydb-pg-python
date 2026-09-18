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
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from langchain_google_alloydb_pg.engine import AlloyDBEngine
from langchain_google_alloydb_pg.tools import (
    AlloyDBIfInput,
    AlloyDBIfTool,
    AlloyDBSentimentTool,
    AlloyDBSummaryTool,
    AlloyDBToolError,
    SentimentInput,
    SummaryInput,
)


@pytest.fixture
def mock_engine():
    """Fixture providing a mock AlloyDBEngine for tools testing."""

    class DummyEngine(AlloyDBEngine):

        def __init__(self):
            self._loop = None

    engine = DummyEngine()

    pool_mock = MagicMock()
    conn_mock = AsyncMock()
    conn_mock.execute.return_value = MagicMock(
        scalar=MagicMock(return_value="positive")
    )
    pool_mock.connect.return_value.__aenter__.return_value = conn_mock
    engine._pool = pool_mock

    async def _async_runner(coro):
        return await coro

    engine._run_as_async = AsyncMock(side_effect=_async_runner)

    def _sync_runner(coro):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

            def in_thread():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            return executor.submit(in_thread).result()

    engine._run_as_sync = MagicMock(side_effect=_sync_runner)
    return engine


def test_tool_metadata(mock_engine):
    """Test tool attributes, schemas, names, and error handling configurations."""
    sentiment_tool = AlloyDBSentimentTool(engine=mock_engine)
    assert sentiment_tool.name == "alloydb_sentiment_tool"
    assert "sentiment" in sentiment_tool.description.lower()
    assert sentiment_tool.args_schema == SentimentInput
    assert sentiment_tool.handle_tool_error is True
    assert sentiment_tool.model_id is None

    summary_tool = AlloyDBSummaryTool(engine=mock_engine)
    assert summary_tool.name == "alloydb_summary_tool"
    assert "summarize" in summary_tool.description.lower()
    assert summary_tool.args_schema == SummaryInput
    assert summary_tool.handle_tool_error is True
    assert summary_tool.model_id is None
    assert summary_tool.additional_instructions is None

    if_tool = AlloyDBIfTool(engine=mock_engine)
    assert if_tool.name == "alloydb_if_tool"
    assert "semantic condition" in if_tool.description.lower()
    assert if_tool.args_schema == AlloyDBIfInput
    assert if_tool.handle_tool_error is True
    assert if_tool.model_id is None


@pytest.mark.asyncio
async def test_sentiment_tool_arun_happy_path(mock_engine):
    """Test that AlloyDBSentimentTool._arun executes sentiment analysis SQL asynchronously."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "positive"

    tool = AlloyDBSentimentTool(engine=mock_engine)
    result = await tool._arun("I really love this product!")
    assert result == "positive"
    assert mock_engine._run_as_async.called

    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT google_ml.analyze_sentiment(:content)" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {"content": "I really love this product!"}


def test_sentiment_tool_run_happy_path(mock_engine):
    """Test that AlloyDBSentimentTool._run executes sentiment analysis SQL synchronously."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "positive"

    tool = AlloyDBSentimentTool(engine=mock_engine)
    result = tool._run("I really love this product!")
    assert result == "positive"
    assert mock_engine._run_as_sync.called
    assert conn_mock.execute.called


@pytest.mark.asyncio
async def test_sentiment_tool_custom_model_id(mock_engine):
    """Test custom model_id in AlloyDBSentimentTool via init and invocation."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "neutral"

    # Test via initialization parameter
    tool = AlloyDBSentimentTool(engine=mock_engine, model_id="fin-sentiment")
    result = await tool._arun("Quarterly earnings were flat.")
    assert result == "neutral"
    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT google_ml.analyze_sentiment(:content, :model_id)" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {
        "content": "Quarterly earnings were flat.",
        "model_id": "fin-sentiment",
    }

    # Test via runtime parameter override
    tool_default = AlloyDBSentimentTool(engine=mock_engine)
    result2 = await tool_default._arun(
        "Quarterly earnings were flat.", model_id="custom-model-2"
    )
    assert result2 == "neutral"
    executed_params2 = conn_mock.execute.call_args[0][1]
    assert executed_params2["model_id"] == "custom-model-2"


@pytest.mark.asyncio
async def test_sentiment_tool_null_raises_error(mock_engine):
    """Test that AlloyDBSentimentTool raises AlloyDBToolError when SQL returns NULL."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = None

    tool = AlloyDBSentimentTool(engine=mock_engine)
    with pytest.raises(AlloyDBToolError, match="returned NULL"):
        await tool._arun("Ambiguous input")

    with pytest.raises(ValueError, match="returned NULL"):
        tool._run("Ambiguous input")


@pytest.mark.asyncio
async def test_summary_tool_arun_happy_path(mock_engine):
    """Test that AlloyDBSummaryTool._arun executes text summarization SQL asynchronously."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "Brief summary."

    tool = AlloyDBSummaryTool(engine=mock_engine)
    result = await tool._arun("This is a very long document...")
    assert result == "Brief summary."
    assert mock_engine._run_as_async.called

    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT google_ml.summarize(:content)" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {"content": "This is a very long document..."}


def test_summary_tool_run_happy_path(mock_engine):
    """Test that AlloyDBSummaryTool._run executes text summarization SQL synchronously."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "Sync summary."

    tool = AlloyDBSummaryTool(engine=mock_engine)
    result = tool._run("This is a document.")
    assert result == "Sync summary."
    assert mock_engine._run_as_sync.called
    assert conn_mock.execute.called


@pytest.mark.asyncio
async def test_summary_tool_additional_instructions(mock_engine):
    """Test AlloyDBSummaryTool with additional_instructions parameter."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "Bulleted summary."

    tool = AlloyDBSummaryTool(
        engine=mock_engine, additional_instructions="In 3 bullet points"
    )
    result = await tool._arun("Long article content...")
    assert result == "Bulleted summary."

    executed_query = conn_mock.execute.call_args[0][0].text
    assert (
        "SELECT google_ml.summarize(:content, :additional_instructions)"
        in executed_query
    )
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {
        "content": "Long article content...",
        "additional_instructions": "In 3 bullet points",
    }


@pytest.mark.asyncio
async def test_summary_tool_model_and_instructions(mock_engine):
    """Test AlloyDBSummaryTool with both model_id and additional_instructions."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "Custom summary."

    tool = AlloyDBSummaryTool(engine=mock_engine)
    result = await tool._arun(
        "Article content...",
        additional_instructions="In 1 sentence",
        model_id="gemini-pro-summary",
    )
    assert result == "Custom summary."

    executed_query = conn_mock.execute.call_args[0][0].text
    assert (
        "SELECT google_ml.summarize(:content, :additional_instructions,"
        " :model_id)" in executed_query
    )
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {
        "content": "Article content...",
        "additional_instructions": "In 1 sentence",
        "model_id": "gemini-pro-summary",
    }


@pytest.mark.asyncio
async def test_summary_tool_null_raises_error(mock_engine):
    """Test that AlloyDBSummaryTool raises AlloyDBToolError when SQL returns NULL."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = None

    tool = AlloyDBSummaryTool(engine=mock_engine)
    with pytest.raises(AlloyDBToolError, match="returned NULL"):
        await tool._arun("Ambiguous input causing NULL")

    with pytest.raises(ValueError, match="returned NULL"):
        tool._run("Ambiguous input causing NULL")


@pytest.mark.asyncio
async def test_if_tool_arun_true_and_false(mock_engine):
    """Test AlloyDBIfTool._arun executes boolean condition SQL returning True and False."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    tool = AlloyDBIfTool(engine=mock_engine)

    conn_mock.execute.return_value.scalar.return_value = True
    res_true = await tool._arun("Is the sky blue?")
    assert res_true is True

    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT google_ml.if(:condition)" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {"condition": "Is the sky blue?"}

    conn_mock.execute.return_value.scalar.return_value = False
    res_false = await tool._arun("Is grass purple?")
    assert res_false is False


def test_if_tool_run_sync(mock_engine):
    """Test AlloyDBIfTool._run executes synchronously returning True and False."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    tool = AlloyDBIfTool(engine=mock_engine)

    conn_mock.execute.return_value.scalar.return_value = True
    assert tool._run("Is 2+2=4?") is True
    assert mock_engine._run_as_sync.called

    conn_mock.execute.return_value.scalar.return_value = False
    assert tool._run("Is 2+2=5?") is False


@pytest.mark.asyncio
async def test_if_tool_custom_model_id(mock_engine):
    """Test AlloyDBIfTool with custom model_id."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = True

    tool = AlloyDBIfTool(engine=mock_engine, model_id="gemini-classifier")
    result = await tool._arun("Is the sentiment positive?")
    assert result is True

    executed_query = conn_mock.execute.call_args[0][0].text
    assert "SELECT google_ml.if(:condition, :model_id)" in executed_query
    executed_params = conn_mock.execute.call_args[0][1]
    assert executed_params == {
        "condition": "Is the sentiment positive?",
        "model_id": "gemini-classifier",
    }


@pytest.mark.asyncio
async def test_if_tool_null_raises_error(mock_engine):
    """Test that AlloyDBIfTool raises AlloyDBToolError on SQL NULL and never silently returns False."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = None

    tool = AlloyDBIfTool(engine=mock_engine)
    with pytest.raises(AlloyDBToolError, match="returned NULL"):
        await tool._arun("Ambiguous condition")

    with pytest.raises(ValueError, match="returned NULL"):
        tool._run("Ambiguous condition")


def test_pydantic_schema_validation_empty_string():
    """Test that empty strings are rejected by Pydantic validation."""
    with pytest.raises(ValidationError):
        SentimentInput(content="")

    with pytest.raises(ValidationError):
        SummaryInput(content="")

    with pytest.raises(ValidationError):
        AlloyDBIfInput(condition="")


def test_pydantic_schema_validation_whitespace():
    """Test that whitespace-only strings are rejected by Pydantic validation."""
    with pytest.raises(ValidationError):
        SentimentInput(content="   \n\t  ")

    with pytest.raises(ValidationError):
        SummaryInput(content="   ")

    with pytest.raises(ValidationError):
        AlloyDBIfInput(condition="   ")

    with pytest.raises(ValidationError):
        SentimentInput(content="valid", model_id="   ")

    with pytest.raises(ValidationError):
        SummaryInput(content="valid", additional_instructions="   ")

    with pytest.raises(ValidationError):
        SummaryInput(content="valid", model_id="   ")

    with pytest.raises(ValidationError):
        AlloyDBIfInput(condition="valid", model_id="   ")


def test_pydantic_schema_whitespace_stripping():
    """Test that leading/trailing whitespace is stripped by Pydantic validators."""
    s = SentimentInput(content="  positive review  ", model_id="  custom-model  ")
    assert s.content == "positive review"
    assert s.model_id == "custom-model"

    sum_input = SummaryInput(
        content="  article  ",
        additional_instructions="  bullets  ",
        model_id="  mod  ",
    )
    assert sum_input.content == "article"
    assert sum_input.additional_instructions == "bullets"
    assert sum_input.model_id == "mod"

    if_input = AlloyDBIfInput(condition="  is valid?  ", model_id="  mod  ")
    assert if_input.condition == "is valid?"
    assert if_input.model_id == "mod"


@pytest.mark.asyncio
async def test_concurrency_cross_loop_safety():
    """Test that _arun routes execution via _run_as_async to preserve cross-loop safety."""
    engine_loop = asyncio.new_event_loop()
    try:

        class CrossLoopSafeEngine(AlloyDBEngine):

            def __init__(self, loop):
                self._loop = loop
                self._pool = MagicMock()
                conn_mock = AsyncMock()
                conn_mock.execute.return_value = MagicMock(
                    scalar=MagicMock(return_value="positive")
                )
                self._pool.connect.return_value.__aenter__.return_value = conn_mock

            async def _run_as_async(self, coro):
                return await coro

        engine = CrossLoopSafeEngine(engine_loop)
        engine._run_as_async = AsyncMock(side_effect=engine._run_as_async)
        tool = AlloyDBSentimentTool(engine=engine)

        result = await tool._arun("Cross-loop check")
        assert result == "positive"
        assert engine._run_as_async.called
    finally:
        engine_loop.close()


def test_deadlock_guard_in_run():
    """Test that calling _run from within the engine's background loop raises RuntimeError."""
    engine_loop = asyncio.new_event_loop()

    class DeadlockEngine(AlloyDBEngine):

        def __init__(self, loop):
            self._loop = loop
            self._pool = MagicMock()

    engine = DeadlockEngine(engine_loop)
    tool = AlloyDBSentimentTool(engine=engine)

    async def caller_in_loop():
        return tool._run("test deadlock condition")

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Cannot call synchronous '_run' from the engine's background"
                " event loop"
            ),
        ):
            engine_loop.run_until_complete(caller_in_loop())
    finally:
        engine_loop.close()


def test_run_manager_callback_invocation_sync(mock_engine):
    """Test that _run invokes run_manager lifecycle callbacks."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "positive"

    tool = AlloyDBSentimentTool(engine=mock_engine)
    mock_run_manager = MagicMock()
    result = tool._run("test with run manager", run_manager=mock_run_manager)
    assert result == "positive"
    assert mock_run_manager.on_tool_start.called is True
    assert mock_run_manager.on_tool_end.called is True

    # Test error callback
    conn_mock.execute.return_value.scalar.return_value = None
    mock_err_manager = MagicMock()
    with pytest.raises(ValueError):
        tool._run("test causing error", run_manager=mock_err_manager)
    assert mock_err_manager.on_tool_error.called is True


@pytest.mark.asyncio
async def test_run_manager_callback_invocation_async(mock_engine):
    """Test that _arun invokes run_manager lifecycle callbacks."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = "positive"

    tool = AlloyDBSentimentTool(engine=mock_engine)
    mock_async_manager = MagicMock()
    result = await tool._arun("test with async manager", run_manager=mock_async_manager)
    assert result == "positive"
    assert mock_async_manager.on_tool_start.called is True
    assert mock_async_manager.on_tool_end.called is True

    # Test error callback
    conn_mock.execute.return_value.scalar.return_value = None
    mock_err_manager = MagicMock()
    with pytest.raises(ValueError):
        await tool._arun("test causing error", run_manager=mock_err_manager)
    assert mock_err_manager.on_tool_error.called is True


@pytest.mark.asyncio
async def test_invoke_and_ainvoke_error_recovery(mock_engine):
    """Test that handle_tool_error recovers from AlloyDBToolError during invoke/ainvoke."""
    conn_mock = mock_engine._pool.connect.return_value.__aenter__.return_value
    conn_mock.execute.return_value.scalar.return_value = None

    tool = AlloyDBSentimentTool(engine=mock_engine)
    sync_result = tool.invoke({"content": "Ambiguous input"})
    assert isinstance(sync_result, str)
    assert "NULL" in sync_result

    async_result = await tool.ainvoke({"content": "Ambiguous input"})
    assert isinstance(async_result, str)
    assert "NULL" in async_result

    if_tool = AlloyDBIfTool(engine=mock_engine)
    if_result = if_tool.invoke({"condition": "Ambiguous condition"})
    assert isinstance(if_result, str)
    assert "NULL" in if_result
