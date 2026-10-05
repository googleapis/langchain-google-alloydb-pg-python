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
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy.exc
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine


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


@pytest.mark.asyncio
class TestEngineForecast:
    """Live tests. They need forecasting enabled on the instance
    (google_ml_integration.enable_forecasting) and fail, not skip, without
    it. The success tests call the ts_forecasting model FORECAST_MODEL_ID
    (default "timesfm") and fail when it is not registered. The error tests
    use a model_id that is never registered, so they check that calls reach
    google_ml.forecast and that its errors reach the caller unchanged."""

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
