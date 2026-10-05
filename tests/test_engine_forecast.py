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
import os
import re
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
import sqlalchemy.exc
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBModelManager


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    # Run on the engine's background loop, like aforecast/forecast do. Pooled
    # asyncpg connections are bound to the event loop that opened them. Using
    # engine._pool directly from the test's loop hands aforecast a connection
    # from another loop, whose BEGIN then fails and whose rollback on close
    # raises "cannot rollback; the transaction is in error state".
    async def run(engine: AlloyDBEngine, query: str) -> None:
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def acall(engine: AlloyDBEngine, api: str, *args: Any, **kwargs: Any) -> Any:
    """Call aforecast directly, or the sync forecast from a worker thread."""
    if api == "forecast":
        return await asyncio.to_thread(engine.forecast, *args, **kwargs)
    return await engine.aforecast(*args, **kwargs)


# The ts_forecasting model the success tests call. It must be registered on
# the instance with google_ml.create_model(model_type => 'ts_forecasting').
FORECAST_MODEL_ID = os.environ.get("FORECAST_MODEL_ID", "timesfm")

# The columns google_ml.forecast returns, in order.
FORECAST_COLUMNS = [
    "forecast_timestamp",
    "forecast_value",
    "confidence_level",
    "prediction_interval_lower_bound",
    "prediction_interval_upper_bound",
    "ai_forecast_status",
]

# The last timestamp in ts_table (2026-01-01 plus 29 days).
TS_TABLE_LAST_TIMESTAMP = datetime(2026, 1, 30)

# The ways to pass ts_table as the source. See ts_table_source_kwargs.
TS_TABLE_SOURCES = ["table_name", "table_name_with_schema_name", "query"]


def ts_table_source_kwargs(source: str, ts_table: str) -> dict[str, Any]:
    """Keyword arguments that read ts_table through the given source."""
    return {
        "table_name": {"table_name": ts_table},
        "table_name_with_schema_name": {
            "table_name": ts_table,
            "schema_name": "public",
        },
        "query": {"query": f'SELECT ts, val FROM "{ts_table}"'},
    }[source]


def assert_forecast(
    results: list[dict],
    *,
    horizon: int,
    conf_level: float,
    last_input_timestamp: datetime,
) -> None:
    """Check the shape of a forecast, not the forecast values themselves.
    The engine returns rows in whatever order the server produces them, and
    SQL does not guarantee one, so the rows are checked in timestamp order."""
    assert len(results) == horizon, results
    assert all(row.get("forecast_timestamp") is not None for row in results), results
    rows = sorted(results, key=lambda row: row["forecast_timestamp"])
    for row in rows:
        assert list(row) == FORECAST_COLUMNS, row
        assert all(row[c] is not None for c in FORECAST_COLUMNS[:5]), row
        assert row["confidence_level"] == pytest.approx(conf_level), row
        assert (
            row["prediction_interval_lower_bound"]
            <= row["forecast_value"]
            <= row["prediction_interval_upper_bound"]
        ), row
    timestamps = [row["forecast_timestamp"] for row in rows]
    assert len(set(timestamps)) == len(timestamps), timestamps
    assert all(ts > last_input_timestamp for ts in timestamps), timestamps


# Errors that would mean the source or its columns could not be resolved.
SOURCE_ERRORS = {"42P01", "3F000", "42602", "42703"}

# Largest value of a PostgreSQL integer (int4) column or parameter.
MAX_INT32 = 2**31 - 1


@pytest.mark.asyncio
class TestEngineForecast:
    """Live tests. They need forecasting enabled on the instance
    (google_ml_integration.enable_forecasting) and fail, not skip, without
    it. The success tests call the ts_forecasting model FORECAST_MODEL_ID
    (default "timesfm") and fail when it is not registered. The error tests
    use a model_id that is never registered, or a model that is not a
    forecasting model, so they check that calls reach google_ml.forecast,
    that it reads the source, and that its errors and client-side
    validation reach the caller unchanged."""

    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def ts_table(self, engine):
        table = "forecast_live_ts_" + uuid.uuid4().hex
        await aexecute(
            engine,
            f"""
            CREATE TABLE "{table}" AS
            SELECT timestamp '2026-01-01' + i * interval '1 day' AS ts,
                   (i % 7)::float8 AS val
            FROM generate_series(0, 29) AS i
            """,
        )
        yield table
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table}"')

    @pytest_asyncio.fixture(scope="class")
    async def mixed_case_table(self, engine):
        """A mixed-case table with mixed-case columns in a mixed-case,
        non-public schema. Yields (schema_name, table_name)."""
        suffix = uuid.uuid4().hex
        schema = f"forecast_live_ts_{suffix}_Sch"
        table = f"forecast_live_ts_{suffix}_MiXed"
        await aexecute(engine, f'CREATE SCHEMA "{schema}"')
        try:
            await aexecute(
                engine,
                f"""
                CREATE TABLE "{schema}"."{table}" AS
                SELECT timestamp '2026-01-01' + i * interval '1 day' AS "Ts",
                       (i % 7)::float8 AS "Val"
                FROM generate_series(0, 29) AS i
                """,
            )
            yield schema, table
        finally:
            await aexecute(engine, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

    @pytest_asyncio.fixture(scope="class")
    async def non_forecast_model(self, engine):
        """A registered text_embedding model. google_ml.forecast checks only
        that the model exists before it reads the source, so this gets a call
        to the point where the table or query and the columns are read. It
        fails later because the model is not a forecasting model. Tests using
        it rely on that check order. Update them if an extension upgrade
        changes it."""
        model_id = "forecast_live_model_" + uuid.uuid4().hex
        model_manager = await AlloyDBModelManager.create(engine)
        await model_manager.acreate_model(
            model_id=model_id,
            model_provider="google",
            model_qualified_name="text-embedding-005",
            model_type="text_embedding",
        )
        yield model_id
        await model_manager.adrop_model(model_id)

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize("source", TS_TABLE_SOURCES)
    async def test_forecast(self, engine, ts_table, api, source):
        """Both entry points and every source return a real forecast: one row
        per step with the google_ml.forecast columns, the requested
        confidence level, the forecast inside its prediction interval, and
        distinct timestamps after the last input timestamp."""
        results = await acall(
            engine,
            api,
            FORECAST_MODEL_ID,
            timestamp_column="ts",
            data_column="val",
            horizon=3,
            conf_level=0.8,
            **ts_table_source_kwargs(source, ts_table),
        )
        assert_forecast(
            results,
            horizon=3,
            conf_level=0.8,
            last_input_timestamp=TS_TABLE_LAST_TIMESTAMP,
        )

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize("source", TS_TABLE_SOURCES)
    async def test_forecast_server_error_reaches_caller(
        self, engine, ts_table, api, source
    ):
        """Both entry points and every source reach google_ml.forecast, and its
        own error (P0001 for an unregistered model, which the server checks
        before reading the source) reaches the caller unchanged. The padded
        model_id in the message shows it was bound exactly as given, without
        stripping. This relies on the server checking the model first. Update
        the test if an extension upgrade changes that order."""
        model_id = "  langchain_missing_model_" + uuid.uuid4().hex + "  "
        with pytest.raises(sqlalchemy.exc.DBAPIError) as exc_info:
            await acall(
                engine,
                api,
                model_id,
                timestamp_column="ts",
                data_column="val",
                horizon=3,
                conf_level=0.8,
                **ts_table_source_kwargs(source, ts_table),
            )
        orig = exc_info.value.orig
        assert getattr(orig, "sqlstate", None) == "P0001", exc_info.value
        assert f"Model does not exist for model_id: {model_id}" in str(orig)

    def source_kwargs(
        self, source: str, ts_table: str, mixed_case_table: tuple[str, str]
    ) -> dict[str, Any]:
        schema, table = mixed_case_table
        return {
            "table_name": {
                "table_name": ts_table,
                "timestamp_column": "ts",
                "data_column": "val",
            },
            "mixed_case_schema_and_table": {
                "table_name": table,
                "schema_name": schema,
                "timestamp_column": "Ts",
                "data_column": "Val",
            },
            "query": {
                "query": f'SELECT "Ts", "Val" FROM "{schema}"."{table}"',
                "timestamp_column": "Ts",
                "data_column": "Val",
            },
        }[source]

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize(
        "source", ["table_name", "mixed_case_schema_and_table", "query"]
    )
    async def test_forecast_reads_source(
        self, engine, ts_table, mixed_case_table, non_forecast_model, api, source
    ):
        """Every source is read: the quoted "schema"."table" (including a
        mixed-case table in a mixed-case schema) or the query, and the
        columns. The call still fails because the model is not a forecasting
        model, but not with a missing relation, schema or column error."""
        with pytest.raises(sqlalchemy.exc.DBAPIError) as exc_info:
            await acall(
                engine,
                api,
                non_forecast_model,
                horizon=3,
                conf_level=0.8,
                **self.source_kwargs(source, ts_table, mixed_case_table),
            )
        orig = exc_info.value.orig
        assert getattr(orig, "sqlstate", None) not in SOURCE_ERRORS, orig
        assert "forecast_live_ts_" not in str(orig), orig
        assert "Model does not exist" not in str(orig), orig

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize(
        "case, sqlstate",
        [
            ("wrong_schema", "42P01"),
            ("missing_table", "42P01"),
            ("wrong_case_column", "42703"),
            ("padded_column", "42703"),
        ],
    )
    async def test_forecast_source_error_reaches_caller(
        self, engine, mixed_case_table, non_forecast_model, api, case, sqlstate
    ):
        """Missing tables and columns are the server's own errors, raised
        unchanged. This is also the control for test_forecast_reads_source:
        schema_name matters, and names are used exactly as given, neither
        case-folded nor stripped (we quote the schema and table, and the
        server quotes the columns)."""
        schema, table = mixed_case_table
        kwargs: dict[str, Any] = {
            "wrong_schema": {"table_name": table},
            "missing_table": {"table_name": table + "_x", "schema_name": schema},
            "wrong_case_column": {
                "table_name": table,
                "schema_name": schema,
                "data_column": "val",
            },
            "padded_column": {
                "table_name": table,
                "schema_name": schema,
                "timestamp_column": " Ts ",
            },
        }[case]
        kwargs = {"timestamp_column": "Ts", "data_column": "Val", **kwargs}
        with pytest.raises(sqlalchemy.exc.DBAPIError) as exc_info:
            await acall(
                engine, api, non_forecast_model, horizon=3, conf_level=0.8, **kwargs
            )
        orig = exc_info.value.orig
        assert getattr(orig, "sqlstate", None) == sqlstate, orig
        assert "does not exist" in str(orig), orig

    @pytest.mark.parametrize(
        "horizon, conf_level",
        [
            (1, 1e-9),
            (128, 0.999),
            (np.int64(7), np.float32(0.5)),
        ],
    )
    async def test_forecast_boundary_values_reach_server(
        self, engine, ts_table, horizon, conf_level
    ):
        """The horizon and conf_level boundaries pass client-side validation
        and are accepted by the driver. The server then reports the
        unregistered model."""
        model_id = "langchain_missing_model_" + uuid.uuid4().hex
        with pytest.raises(sqlalchemy.exc.DBAPIError) as exc_info:
            await engine.aforecast(
                model_id,
                timestamp_column="ts",
                data_column="val",
                horizon=horizon,
                conf_level=conf_level,
                table_name=ts_table,
            )
        assert f"Model does not exist for model_id: {model_id}" in str(
            exc_info.value.orig
        )

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize(
        "model_id, overrides, message",
        [
            # Exactly one source.
            ("m", {"table_name": None}, "Exactly one of 'table_name' or 'query'"),
            ("m", {"query": "SELECT 1"}, "Exactly one of 'table_name' or 'query'"),
            # Required strings: empty, blank or not a string.
            ("", {}, "model_id must be a non-empty string"),
            ("   ", {}, "model_id must be a non-empty string"),
            (123, {}, "model_id must be a non-empty string"),
            ("m", {"table_name": ""}, "table_name must be a non-empty string"),
            ("m", {"table_name": "   "}, "table_name must be a non-empty string"),
            ("m", {"table_name": 123}, "table_name must be a non-empty string"),
            ("m", {"schema_name": ""}, "schema_name must be a non-empty string"),
            ("m", {"schema_name": None}, "schema_name must be a non-empty string"),
            (
                "m",
                {"table_name": None, "query": ""},
                "query must be a non-empty string",
            ),
            (
                "m",
                {"table_name": None, "query": "   "},
                "query must be a non-empty string",
            ),
            (
                "m",
                {"table_name": None, "query": 123},
                "query must be a non-empty string",
            ),
            (
                "m",
                {"timestamp_column": ""},
                "timestamp_column must be a non-empty string",
            ),
            (
                "m",
                {"timestamp_column": None},
                "timestamp_column must be a non-empty string",
            ),
            ("m", {"data_column": ""}, "data_column must be a non-empty string"),
            ("m", {"data_column": 1}, "data_column must be a non-empty string"),
            # horizon outside 1..128, including one past the int4 maximum.
            *(
                ("m", {"horizon": bad}, "horizon must be between 1 and 128")
                for bad in (0, -5, 129, MAX_INT32 + 1)
            ),
            # conf_level outside (0, 1), including NaN and infinities.
            *(
                (
                    "m",
                    {"conf_level": bad},
                    "conf_level must be strictly between 0 and 1",
                )
                for bad in (
                    0,
                    1.0,
                    -0.5,
                    1.5,
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                )
            ),
        ],
    )
    async def test_forecast_validation_error(
        self, engine, ts_table, api, model_id, overrides, message
    ):
        """Invalid arguments raise ValueError on a real engine before any SQL,
        and the engine keeps working afterwards."""
        kwargs: dict[str, Any] = dict(
            timestamp_column="ts",
            data_column="val",
            horizon=3,
            conf_level=0.8,
            table_name=ts_table,
        )
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=re.escape(message)):
            await acall(engine, api, model_id, **kwargs)
        await aexecute(engine, "SELECT 1")


APIS = ["_aforecast", "aforecast", "forecast"]

# Valid keyword arguments after model_id. Tests override single entries.
VALID_KWARGS: dict[str, Any] = dict(
    timestamp_column="ts",
    data_column="val",
    horizon=3,
    conf_level=0.8,
    table_name="t",
)


def forecast_signature(api: str) -> inspect.Signature:
    """The signature of AlloyDBEngine.<api> without ``self``."""
    sig = inspect.signature(getattr(AlloyDBEngine, api))
    return sig.replace(parameters=list(sig.parameters.values())[1:])


@pytest.mark.parametrize("api", APIS)
def test_forecast_signature(api):
    """model_id may be positional. Everything after it is keyword-only.
    conf_level has no default, while schema_name defaults to "public" and the
    sources to None."""
    params = forecast_signature(api).parameters
    assert list(params) == [
        "model_id",
        "timestamp_column",
        "data_column",
        "horizon",
        "conf_level",
        "table_name",
        "schema_name",
        "query",
    ]
    assert params["model_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for name, p in params.items()
        if name != "model_id"
    )
    for name in ("timestamp_column", "data_column", "horizon", "conf_level"):
        assert params[name].default is inspect.Parameter.empty
    assert params["table_name"].default is None
    assert params["schema_name"].default == "public"
    assert params["query"].default is None


@pytest.mark.parametrize("api", APIS)
def test_forecast_rejects_positional_arguments(api):
    """The arguments after model_id cannot be passed positionally."""
    with pytest.raises(TypeError, match="positional argument"):
        forecast_signature(api).bind("m", "ts", "val", 4, 0.9, "t")


@pytest.mark.parametrize("api", APIS)
def test_forecast_conf_level_required(api):
    """Omitting conf_level is a TypeError from the call itself."""
    kwargs = {k: v for k, v in VALID_KWARGS.items() if k != "conf_level"}
    with pytest.raises(TypeError, match="conf_level"):
        forecast_signature(api).bind("m", **kwargs)
