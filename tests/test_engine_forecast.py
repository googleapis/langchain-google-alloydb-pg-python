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
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy.exc
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    # Run on the engine's background loop, like aforecast/forecast do. Pooled
    # asyncpg connections are bound to the event loop that opened them; using
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


def skip_if_forecasting_disabled(error: sqlalchemy.exc.DBAPIError) -> None:
    # google_ml.forecast checks the google_ml_integration.enable_forecasting
    # flag before anything else. Check the server message only: str(error)
    # also contains the SQL and its parameters.
    if "enable_forecasting" in str(error.orig):
        pytest.skip(f"Forecasting is disabled on the instance: {error.orig}")


@pytest.mark.asyncio
class TestEngineForecast:
    """Live tests. No ts_forecasting model is registered on the test instance,
    so they check that calls reach google_ml.forecast and that its errors
    reach the caller; the real forecast test skips until a model is
    registered."""

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

    async def test_forecast(self, engine, ts_table):
        """A real forecast; skips while no forecast model is registered."""
        model_id = os.environ.get("FORECAST_MODEL_ID", "test_model")
        try:
            results = await engine.aforecast(
                model_id,
                timestamp_column="ts",
                data_column="val",
                horizon=3,
                conf_level=0.8,
                table_name=ts_table,
            )
        except sqlalchemy.exc.DBAPIError as e:
            skip_if_forecasting_disabled(e)
            if f"Model does not exist for model_id: {model_id}" in str(e.orig):
                pytest.skip(f"Forecast model {model_id!r} is not registered: {e.orig}")
            raise
        assert len(results) == 3
        assert {"forecast_timestamp", "forecast_value"} <= set(results[0])

    @pytest.mark.parametrize("api", ["aforecast", "forecast"])
    @pytest.mark.parametrize(
        "source",
        [
            "table_name",
            "table_name_with_schema_name",
            "query",
        ],
    )
    async def test_forecast_server_error_reaches_caller(
        self, engine, ts_table, api, source
    ):
        """Both entry points and every source reach google_ml.forecast, and its
        own error (P0001 for an unregistered model, which the server checks
        before reading the source) reaches the caller unchanged. The padded
        model_id in the message shows it was bound exactly as given, without
        stripping. This relies on the server checking the model first; update
        the test if an extension upgrade changes that order."""
        model_id = "  langchain_missing_model_" + uuid.uuid4().hex + "  "
        source_kwargs: dict[str, Any] = {
            "table_name": {"table_name": ts_table},
            "table_name_with_schema_name": {
                "table_name": ts_table,
                "schema_name": "public",
            },
            "query": {"query": f'SELECT ts, val FROM "{ts_table}"'},
        }[source]
        with pytest.raises(sqlalchemy.exc.DBAPIError) as exc_info:
            await acall(
                engine,
                api,
                model_id,
                timestamp_column="ts",
                data_column="val",
                horizon=3,
                conf_level=0.8,
                **source_kwargs,
            )
        skip_if_forecasting_disabled(exc_info.value)
        orig = exc_info.value.orig
        assert getattr(orig, "sqlstate", None) == "P0001", exc_info.value
        assert f"Model does not exist for model_id: {model_id}" in str(orig)
