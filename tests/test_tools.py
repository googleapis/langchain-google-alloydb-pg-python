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
import os
import uuid
from typing import Any, Optional

import pytest
import pytest_asyncio
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBIfTool,
    AlloyDBSentimentTool,
    AlloyDBSummaryTool,
    AlloyDBToolError,
)
from langchain_google_alloydb_pg.tools import (
    AlloyDBIfInput,
    SentimentInput,
    SummaryInput,
    _afetch_scalar,
)

MODEL_ID = "gemini-2.5-flash"
UNKNOWN_MODEL_ID = "no_such_model_" + "x" * 8
# SQLSTATE AlloyDB AI returns when Vertex AI can't find the model (HTTP 404),
# for example because it isn't available in this project or region.
MODEL_NOT_FOUND_SQLSTATE = "GAV05"
# SQLSTATEs meaning an AlloyDB AI function can't be used on this instance.
UNAVAILABLE_SQLSTATES = {
    "42883": "the function does not exist (google_ml_integration is too old)",
    "0A000": "the function is not supported on this PostgreSQL version",
    "GAV07": "the AlloyDB service agent may not call Vertex AI",
}
# AlloyDB AI functions raise (SQLSTATE P0001) unless these flags are on.
AI_QUERY_ENGINE_FLAG = "google_ml_integration.enable_ai_query_engine"
PREVIEW_AI_FUNCTIONS_FLAG = "google_ml_integration.enable_preview_ai_functions"
# Registering a model needs this flag.
MODEL_SUPPORT_FLAG = "google_ml_integration.enable_model_support"
# Names the model the AI functions use when model_id is None.
DEFAULT_LLM_MODEL_FLAG = "google_ml_integration.default_llm_model"


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest_asyncio.fixture(scope="module")
async def engine():
    engine = await AlloyDBEngine.afrom_instance(
        project_id=get_env_var("PROJECT_ID", "project id for google cloud"),
        region=get_env_var("REGION", "region for AlloyDB instance"),
        cluster=get_env_var("CLUSTER_ID", "cluster for AlloyDB"),
        instance=get_env_var("INSTANCE_ID", "instance for AlloyDB"),
        database=get_env_var("DATABASE_ID", "database name on AlloyDB instance"),
    )
    yield engine
    await engine.close()


async def run_sql(
    engine: AlloyDBEngine, query: str, params: Optional[dict[str, Any]] = None
) -> Any:
    """Runs ``query`` and commits. Returns its first value, if it has one."""

    async def run() -> Any:
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query), params or {})
            value = result.scalar() if result.returns_rows else None
            await conn.commit()
        return value

    return await engine._run_as_async(run())


async def disabled_flags(engine: AlloyDBEngine, flags: list[str]) -> list[str]:
    """Returns the database flags in ``flags`` that aren't on."""
    query = "SELECT current_setting(:flag, true)"
    return [f for f in flags if await run_sql(engine, query, {"flag": f}) != "on"]


@pytest_asyncio.fixture(scope="module")
async def null_model_id(engine):
    """Registers an LLM model whose replies AlloyDB AI reads as NULL.

    The model calls the database's default Gemini model, but its output
    transform returns NULL, so any AI function that uses it returns NULL.
    The model and the transform function are dropped afterwards.
    """
    if await disabled_flags(engine, [MODEL_SUPPORT_FLAG]):
        pytest.skip(
            f"Can't register a model on this instance: {MODEL_SUPPORT_FLAG} is off."
        )
    base_model = await run_sql(
        engine, "SELECT current_setting(:flag)", {"flag": DEFAULT_LLM_MODEL_FLAG}
    )
    suffix = uuid.uuid4().hex[:8]
    model_id, transform = f"null_llm_{suffix}", f"public.null_llm_output_{suffix}"
    await run_sql(
        engine,
        f"CREATE FUNCTION {transform}(model_id VARCHAR, response JSON)"
        " RETURNS TEXT LANGUAGE sql AS 'SELECT NULL::text'",
    )
    try:
        await run_sql(
            engine,
            f"""CALL google_ml.create_model(
                model_id => '{model_id}',
                model_request_url => 'publishers/google/models/{base_model}:generateContent',
                model_provider => 'google',
                model_type => 'llm',
                model_qualified_name => '{base_model}',
                model_in_transform_fn => 'google_ml.gemini_llm_input_transform',
                model_out_transform_fn => '{transform}')""",
        )
        try:
            yield model_id
        finally:
            await run_sql(engine, f"CALL google_ml.drop_model('{model_id}')")
    finally:
        await run_sql(engine, f"DROP FUNCTION {transform}(VARCHAR, JSON)")


async def ainvoke_with_model_id(tool: BaseTool, tool_input: dict[str, Any]) -> Any:
    """Calls a tool created with MODEL_ID, or skips if the instance can't use it."""
    try:
        return await tool.ainvoke(tool_input)
    except DBAPIError as e:
        if getattr(e.orig, "sqlstate", None) != MODEL_NOT_FOUND_SQLSTATE:
            raise
        pytest.skip(
            f"{MODEL_ID} is not available on this instance (SQLSTATE "
            f"{MODEL_NOT_FOUND_SQLSTATE}): {e.orig}"
        )


async def available_tool(
    engine: AlloyDBEngine,
    tool: BaseTool,
    tool_input: dict[str, Any],
    flags: list[str],
) -> BaseTool:
    """Returns the tool, or skips the test if its AI function is unavailable.

    ``flags`` are the database flags the tool's AI function requires.
    """
    try:
        await tool.ainvoke(tool_input)
    except DBAPIError as e:
        sqlstate = getattr(e.orig, "sqlstate", None)
        if sqlstate in UNAVAILABLE_SQLSTATES:
            pytest.skip(
                f"{tool.name} is unavailable on this instance (SQLSTATE "
                f"{sqlstate}): {UNAVAILABLE_SQLSTATES[sqlstate]}."
            )
        off = await disabled_flags(engine, flags) if sqlstate == "P0001" else []
        if off:
            pytest.skip(
                f"{tool.name} is disabled on this instance (SQLSTATE P0001): "
                f"{', '.join(off)} is off."
            )
        raise
    return tool


class ToolEventRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("start")

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("end")

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("error")


@pytest.mark.asyncio
class TestAlloyDBToolHelpers:
    async def test_null_result_raises_tool_error(self, engine):
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            await engine._run_as_async(
                _afetch_scalar(engine, "SELECT NULL::text", {}, "returned NULL")
            )
        assert issubclass(AlloyDBToolError, ValueError)

    async def test_sync_call_on_engine_loop_raises(self, engine):
        tool = AlloyDBSentimentTool(engine=engine)

        async def invoke_on_engine_loop() -> Any:
            return tool.invoke({"content": "I love this product!"})

        with pytest.raises(RuntimeError, match="Use 'ainvoke' instead"):
            await engine._run_as_async(invoke_on_engine_loop())


@pytest.mark.asyncio
class TestAlloyDBSentimentTool:
    @pytest_asyncio.fixture(scope="module")
    async def sentiment_tool(self, engine):
        return await available_tool(
            engine,
            AlloyDBSentimentTool(engine=engine),
            {"content": "I love it."},
            [AI_QUERY_ENGINE_FLAG, PREVIEW_AI_FUNCTIONS_FLAG],
        )

    async def test_metadata(self, engine):
        tool = AlloyDBSentimentTool(engine=engine)
        assert tool.name == "alloydb_sentiment_tool"
        assert "sentiment" in tool.description.lower()
        assert tool.args_schema is SentimentInput
        # The agent supplies only the content; model_id is set on the tool.
        assert list(tool.args) == ["content"]
        assert tool.handle_tool_error is False
        assert tool.model_id is None

    async def test_invoke(self, sentiment_tool):
        assert (
            sentiment_tool.invoke({"content": "I really love this product!"})
            == "positive"
        )
        # From a plain thread, with no running event loop.
        result = await asyncio.to_thread(
            sentiment_tool.invoke, {"content": "I really love this product!"}
        )
        assert result == "positive"

    async def test_ainvoke(self, sentiment_tool):
        result = await sentiment_tool.ainvoke(
            {"content": "This is terrible, I hate it and want a refund."}
        )
        assert result == "negative"

    async def test_model_id(self, engine, sentiment_tool):
        tool = AlloyDBSentimentTool(engine=engine, model_id=MODEL_ID)
        content = {"content": "I really love this product!"}
        assert await ainvoke_with_model_id(tool, content) == "positive"

    async def test_unknown_model_id_raises(self, engine, sentiment_tool):
        tool = AlloyDBSentimentTool(engine=engine, model_id=UNKNOWN_MODEL_ID)
        # model_id isn't an agent input, so a model_id in the input is ignored.
        with pytest.raises(DBAPIError):
            await tool.ainvoke({"content": "I love it.", "model_id": MODEL_ID})

    async def test_null_result_raises(self, engine, sentiment_tool, null_model_id):
        tool = AlloyDBSentimentTool(engine=engine, model_id=null_model_id)
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            await tool.ainvoke({"content": "I love it."})
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            tool.invoke({"content": "I love it."})
        # With handle_tool_error=True, the agent gets the message instead.
        tool = AlloyDBSentimentTool(
            engine=engine, model_id=null_model_id, handle_tool_error=True
        )
        assert "returned NULL" in await tool.ainvoke({"content": "I love it."})

    async def test_callbacks_fire_once(self, sentiment_tool):
        recorder = ToolEventRecorder()
        sentiment_tool.invoke({"content": "I love it."}, {"callbacks": [recorder]})
        assert recorder.events == ["start", "end"]

    async def test_input_validation(self, engine):
        tool = AlloyDBSentimentTool(engine=engine)
        with pytest.raises(ValidationError, match="content must be a non-empty"):
            await tool.ainvoke({"content": "   \n\t  "})
        with pytest.raises(ValidationError):
            SentimentInput(content="")
        with pytest.raises(ValidationError, match="model_id must be a non-empty"):
            AlloyDBSentimentTool(engine=engine, model_id="   ")
        # Values reach the database as given.
        assert SentimentInput(content="  good  ").content == "  good  "
        assert AlloyDBSentimentTool(engine=engine, model_id=" m ").model_id == " m "


@pytest.mark.asyncio
class TestAlloyDBSummaryTool:
    CONTENT = (
        "AlloyDB for PostgreSQL is a fully managed, PostgreSQL-compatible "
        "database service. It combines Google's storage and compute "
        "technology to deliver high performance for transactional and "
        "analytical workloads, and includes AlloyDB AI for calling machine "
        "learning models directly from SQL."
    )

    @pytest_asyncio.fixture(scope="module")
    async def summary_tool(self, engine):
        return await available_tool(
            engine,
            AlloyDBSummaryTool(engine=engine),
            {"content": self.CONTENT},
            [AI_QUERY_ENGINE_FLAG, PREVIEW_AI_FUNCTIONS_FLAG],
        )

    def check_summary(self, summary: Any) -> None:
        assert isinstance(summary, str)
        assert summary.strip()

    async def test_metadata(self, engine):
        tool = AlloyDBSummaryTool(engine=engine)
        assert tool.name == "alloydb_summary_tool"
        assert "summarize" in tool.description.lower()
        assert tool.args_schema is SummaryInput
        assert list(tool.args) == ["content", "additional_instructions"]
        assert tool.handle_tool_error is False
        assert tool.model_id is None
        assert tool.additional_instructions is None

    async def test_invoke(self, summary_tool):
        self.check_summary(summary_tool.invoke({"content": self.CONTENT}))

    async def test_ainvoke(self, summary_tool):
        self.check_summary(await summary_tool.ainvoke({"content": self.CONTENT}))

    async def test_additional_instructions(self, engine, summary_tool):
        instructions = "In one short sentence"
        tool = AlloyDBSummaryTool(engine=engine, additional_instructions=instructions)
        self.check_summary(await tool.ainvoke({"content": self.CONTENT}))
        self.check_summary(
            await summary_tool.ainvoke(
                {"content": self.CONTENT, "additional_instructions": instructions}
            )
        )

    async def test_model_id(self, engine, summary_tool):
        tool = AlloyDBSummaryTool(engine=engine, model_id=MODEL_ID)
        self.check_summary(await ainvoke_with_model_id(tool, {"content": self.CONTENT}))
        self.check_summary(
            await ainvoke_with_model_id(
                tool,
                {
                    "content": self.CONTENT,
                    "additional_instructions": "In one short sentence",
                },
            )
        )

    async def test_unknown_model_id_raises(self, engine, summary_tool):
        tool = AlloyDBSummaryTool(engine=engine, model_id=UNKNOWN_MODEL_ID)
        with pytest.raises(DBAPIError):
            await tool.ainvoke({"content": self.CONTENT, "model_id": MODEL_ID})

    async def test_null_result_raises(self, engine, summary_tool, null_model_id):
        tool = AlloyDBSummaryTool(engine=engine, model_id=null_model_id)
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            await tool.ainvoke({"content": self.CONTENT})
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            tool.invoke({"content": self.CONTENT})

    async def test_input_validation(self, engine):
        tool = AlloyDBSummaryTool(engine=engine)
        with pytest.raises(ValidationError, match="content must be a non-empty"):
            await tool.ainvoke({"content": "   "})
        with pytest.raises(ValidationError, match="additional_instructions must"):
            SummaryInput(content="valid", additional_instructions="   ")
        with pytest.raises(ValidationError, match="additional_instructions must"):
            AlloyDBSummaryTool(engine=engine, additional_instructions="   ")
        with pytest.raises(ValidationError, match="model_id must be a non-empty"):
            AlloyDBSummaryTool(engine=engine, model_id="   ")
        parsed = SummaryInput(
            content="  article  ", additional_instructions=" bullets "
        )
        assert (parsed.content, parsed.additional_instructions) == (
            "  article  ",
            " bullets ",
        )


@pytest.mark.asyncio
class TestAlloyDBIfTool:
    TRUE_CONDITION = "Is 2 + 2 equal to 4?"
    FALSE_CONDITION = "Is 2 + 2 equal to 5?"

    @pytest_asyncio.fixture(scope="module")
    async def if_tool(self, engine):
        return await available_tool(
            engine,
            AlloyDBIfTool(engine=engine),
            {"condition": self.TRUE_CONDITION},
            [AI_QUERY_ENGINE_FLAG],
        )

    async def test_metadata(self, engine):
        tool = AlloyDBIfTool(engine=engine)
        assert tool.name == "alloydb_if_tool"
        assert "semantic condition" in tool.description.lower()
        assert tool.args_schema is AlloyDBIfInput
        assert list(tool.args) == ["condition"]
        assert tool.handle_tool_error is False
        assert tool.model_id is None

    async def test_invoke(self, if_tool):
        assert if_tool.invoke({"condition": self.TRUE_CONDITION}) is True
        assert if_tool.invoke({"condition": self.FALSE_CONDITION}) is False

    async def test_ainvoke(self, if_tool):
        assert await if_tool.ainvoke({"condition": self.TRUE_CONDITION}) is True
        assert await if_tool.ainvoke({"condition": self.FALSE_CONDITION}) is False

    async def test_model_id(self, engine, if_tool):
        tool = AlloyDBIfTool(engine=engine, model_id=MODEL_ID)
        condition = {"condition": self.TRUE_CONDITION}
        assert await ainvoke_with_model_id(tool, condition) is True
        condition = {"condition": self.FALSE_CONDITION}
        assert await ainvoke_with_model_id(tool, condition) is False

    async def test_unknown_model_id_raises(self, engine, if_tool):
        tool = AlloyDBIfTool(engine=engine, model_id=UNKNOWN_MODEL_ID)
        with pytest.raises(DBAPIError):
            await tool.ainvoke({"condition": self.TRUE_CONDITION, "model_id": MODEL_ID})

    async def test_null_result_raises(self, engine, if_tool, null_model_id):
        # A NULL result raises instead of being returned as a (truthy) message.
        tool = AlloyDBIfTool(engine=engine, model_id=null_model_id)
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            await tool.ainvoke({"condition": self.TRUE_CONDITION})
        with pytest.raises(AlloyDBToolError, match="returned NULL"):
            tool.invoke({"condition": self.TRUE_CONDITION})

    async def test_input_validation(self, engine):
        tool = AlloyDBIfTool(engine=engine)
        with pytest.raises(ValidationError, match="condition must be a non-empty"):
            await tool.ainvoke({"condition": "   "})
        with pytest.raises(ValidationError):
            AlloyDBIfInput(condition="")
        with pytest.raises(ValidationError, match="model_id must be a non-empty"):
            AlloyDBIfTool(engine=engine, model_id="   ")
        assert AlloyDBIfInput(condition=" is valid? ").condition == " is valid? "
