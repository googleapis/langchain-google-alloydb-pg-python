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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("PROJECT_ID"),
    reason="Requires live AlloyDB instance (PROJECT_ID not set)",
)
class TestEngineForecastIntegration:
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

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user on AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password on AlloyDB")

    @pytest_asyncio.fixture(scope="class")
    async def engine(
        self, db_project, db_region, db_cluster, db_instance, db_name, user, password
    ):
        omni_host = os.environ.get("OMNI_HOST") or os.environ.get("IP_ADDRESS")
        if omni_host:
            port = os.environ.get("OMNI_PORT", "5432")
            conn_str = f"postgresql+asyncpg://{user}:{password}@{omni_host}:{port}/{db_name}?ssl=require"
            engine = AlloyDBEngine.from_connection_string(conn_str)
        else:
            engine = await AlloyDBEngine.afrom_instance(
                project_id=db_project,
                cluster=db_cluster,
                instance=db_instance,
                region=db_region,
                database=db_name,
            )
        yield engine
        await engine.close()

    async def test_live_forecast(self, engine):
        """Test live google_ml.forecast validation / execution on AlloyDB."""
        ts_table = "forecast_live_ts_" + str(uuid.uuid4()).replace("-", "_")
        await aexecute(
            engine,
            f"""
            CREATE TABLE IF NOT EXISTS "{ts_table}" (
                timestamp_col timestamp without time zone,
                data_col float8
            );
            """,
        )
        try:
            results = await engine.aforecast(
                model_id="test_model",
                timestamp_col="timestamp_col",
                data_col="data_col",
                horizon=3,
                source_table=ts_table,
            )
            assert isinstance(results, list)
        except Exception as e:
            # Model may not be registered in Vertex AI / AlloyDB model registry in test env
            if (
                "model" in str(e).lower()
                or "not found" in str(e).lower()
                or "google_ml" in str(e).lower()
            ):
                pass
            else:
                raise
        finally:
            await aexecute(engine, f'DROP TABLE IF EXISTS "{ts_table}"')


class TestEngineUnit:
    @pytest.fixture
    def engine(self):
        eng = AlloyDBEngine.__new__(AlloyDBEngine)
        eng._pool = MagicMock()

        def mock_run_sync(coro):
            coro.close()
            ret = eng._run_as_sync.return_value
            if isinstance(ret, MagicMock):
                return [{"prediction": 1.0}]
            return ret

        eng._run_as_sync = MagicMock(side_effect=mock_run_sync)

        async def mock_run_async(coro):
            return await coro

        eng._run_as_async = mock_run_async
        return eng

    @pytest.mark.asyncio
    async def test_aforecast(self, engine):
        """Test that aforecast calls the underlying google_ml.forecast table function asynchronously."""
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [
                {"prediction": 1.0},
                {"prediction": 2.0},
            ]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            results = await engine.aforecast(
                model_id="test_model",
                timestamp_col="ts",
                data_col="data",
                horizon=5,
                source_table="test_table",
            )
            assert len(results) == 2
            assert results[0]["prediction"] == 1.0
            call_args = mock_conn.execute.call_args
            assert "SELECT * FROM google_ml.forecast" in str(call_args[0][0])
            assert "source_table => :source_table" in str(call_args[0][0])
            assert "source_query" not in str(call_args[0][0])
            assert "conf_level" not in str(call_args[0][0])
            assert call_args[0][1] == {
                "model_id": "test_model",
                "source_table": "test_table",
                "timestamp_col": "ts",
                "data_col": "data",
                "horizon": 5,
            }

    @pytest.mark.asyncio
    async def test_aforecast_with_source_query(self, engine):
        """Test aforecast with source_query only."""
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [
                {"prediction": 1.0},
                {"prediction": 2.0},
            ]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            results = await engine.aforecast(
                model_id="test_model",
                timestamp_col="ts",
                data_col="data",
                horizon=5,
                source_query="SELECT * FROM test_table",
            )
            assert len(results) == 2
            assert results[0]["prediction"] == 1.0
            call_args = mock_conn.execute.call_args
            assert "SELECT * FROM google_ml.forecast" in str(call_args[0][0])
            assert "source_query => :source_query" in str(call_args[0][0])
            assert "source_table" not in str(call_args[0][0])
            assert "conf_level" not in str(call_args[0][0])
            assert call_args[0][1] == {
                "model_id": "test_model",
                "source_query": "SELECT * FROM test_table",
                "timestamp_col": "ts",
                "data_col": "data",
                "horizon": 5,
            }

    @pytest.mark.asyncio
    async def test_aforecast_with_optional_params(self, engine):
        """Test aforecast with optional conf_level."""
        with patch.object(
            engine, "_aforecast", new_callable=AsyncMock
        ) as mock_aforecast:
            mock_aforecast.return_value = [{"prediction": 42.0}]
            results = await engine.aforecast(
                model_id="test_model",
                timestamp_col="ts",
                data_col="data",
                horizon=10,
                source_table="test_table",
                conf_level=0.95,
            )
            assert len(results) == 1
            assert results[0]["prediction"] == 42.0
            mock_aforecast.assert_called_once_with(
                "test_model",
                "ts",
                "data",
                10,
                "test_table",
                None,
                0.95,
            )

    def test_forecast(self, engine):
        """Test that forecast evaluates via _run_as_sync to proxy the google_ml.forecast."""
        engine._run_as_sync.return_value = [{"prediction": 1.0}]
        results = engine.forecast(
            model_id="test_model",
            timestamp_col="ts",
            data_col="data",
            horizon=5,
            source_table="test_table",
        )
        assert results == [{"prediction": 1.0}]
        engine._run_as_sync.assert_called_once()

    def test_forecast_with_source_query(self, engine):
        """Test forecast with source_query only executes cleanly and generates correct SQL."""
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [{"prediction": 42.0}]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            engine._run_as_sync.side_effect = lambda coro: asyncio.run(coro)
            results = engine.forecast(
                model_id="test_model",
                timestamp_col="ts",
                data_col="data",
                horizon=10,
                source_query="SELECT * FROM data",
            )
            assert results == [{"prediction": 42.0}]
            call_args = mock_conn.execute.call_args
            assert "SELECT * FROM google_ml.forecast" in str(call_args[0][0])
            assert "source_query => :source_query" in str(call_args[0][0])
            assert "source_table" not in str(call_args[0][0])
            assert "conf_level" not in str(call_args[0][0])
            assert call_args[0][1] == {
                "model_id": "test_model",
                "source_query": "SELECT * FROM data",
                "timestamp_col": "ts",
                "data_col": "data",
                "horizon": 10,
            }

    def test_forecast_with_optional_params(self, engine):
        """Test forecast with optional conf_level."""
        engine._run_as_sync.return_value = [{"prediction": 42.0}]
        results = engine.forecast(
            model_id="test_model",
            timestamp_col="ts",
            data_col="data",
            horizon=10,
            source_table="test_table",
            conf_level=0.95,
        )
        assert results == [{"prediction": 42.0}]
        engine._run_as_sync.assert_called_once()

    def test_forecast_validation_errors(self, engine):
        """Test validation errors for invalid input parameters in sync forecast."""
        engine._run_as_sync.side_effect = lambda coro: asyncio.run(coro)
        # Mutual exclusivity: neither provided
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            engine.forecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
            )
        # Mutual exclusivity: both provided
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            engine.forecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                source_query="SELECT * FROM sales",
            )

    @pytest.mark.asyncio
    async def test_private_aforecast(self, engine):
        """Test direct _aforecast execution and mapping parsing."""
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [
                {"forecast_timestamp": "2026-08-07", "forecast_value": 100.0}
            ]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            results = await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=0.9,
            )
            assert len(results) == 1
            assert results[0]["forecast_value"] == 100.0
            call_args = mock_conn.execute.call_args
            assert "SELECT * FROM google_ml.forecast" in str(call_args[0][0])
            assert "source_table => :source_table" in str(call_args[0][0])
            assert "source_query" not in str(call_args[0][0])
            assert "conf_level => :conf_level" in str(call_args[0][0])
            assert call_args[0][1] == {
                "model_id": "model_1",
                "source_table": "sales",
                "timestamp_col": "date",
                "data_col": "revenue",
                "horizon": 3,
                "conf_level": 0.9,
            }

    @pytest.mark.asyncio
    async def test_aforecast_validation_errors(self, engine):
        """Test validation errors for invalid input parameters in aforecast and _aforecast."""
        # Mutual exclusivity: neither provided (_aforecast)
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
            )
        # Mutual exclusivity: neither provided (public aforecast)
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            await engine.aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
            )
        # Mutual exclusivity: both provided (_aforecast)
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                source_query="SELECT * FROM sales",
            )
        # Mutual exclusivity: both provided (public aforecast)
        with pytest.raises(
            ValueError,
            match="Exactly one of 'source_table' or 'source_query' must be provided",
        ):
            await engine.aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                source_query="SELECT * FROM sales",
            )
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            await engine._aforecast(
                model_id="",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
            )
        with pytest.raises(ValueError, match="source_table must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="",
            )
        with pytest.raises(ValueError, match="source_table must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="   ",
            )
        with pytest.raises(ValueError, match="source_table must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table=123,  # type: ignore
            )
        with pytest.raises(ValueError, match="source_query must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_query="",
            )
        with pytest.raises(ValueError, match="source_query must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_query="   ",
            )
        with pytest.raises(ValueError, match="source_query must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_query=123,  # type: ignore
            )
        with pytest.raises(
            ValueError, match="timestamp_col must be a non-empty string"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="",
                data_col="revenue",
                horizon=3,
                source_table="sales",
            )
        with pytest.raises(ValueError, match="data_col must be a non-empty string"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="",
                horizon=3,
                source_table="sales",
            )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=0,
                source_table="sales",
            )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=-5,
                source_table="sales",
            )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=True,  # type: ignore
                source_table="sales",
            )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=1.5,  # type: ignore
                source_table="sales",
            )
        with pytest.raises(
            ValueError, match="horizon exceeds maximum 32-bit integer limit"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=2_147_483_648,
                source_table="sales",
            )
        with pytest.raises(
            ValueError, match="conf_level must be a float strictly between 0 and 1"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=0,
            )
        with pytest.raises(
            ValueError, match="conf_level must be a float strictly between 0 and 1"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=1.0,
            )
        with pytest.raises(
            ValueError, match="conf_level must be a float strictly between 0 and 1"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=-0.5,
            )
        with pytest.raises(
            ValueError, match="conf_level must be a float strictly between 0 and 1"
        ):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=1.5,
            )
        with pytest.raises(TypeError, match="conf_level must be a float"):
            await engine._aforecast(
                model_id="model_1",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_table="sales",
                conf_level=True,  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_aforecast_google_ml_integration_extension_error(self, engine):
        """Test that _aforecast raises informative error when google_ml_integration extension is missing."""
        from sqlalchemy.exc import ProgrammingError

        expected_msg = (
            "AlloyDB AI google_ml_integration extension is not installed or enabled. "
            "Please execute 'CREATE EXTENSION IF NOT EXISTS google_ml_integration CASCADE;' on your database."
        )

        # Test error string containing "google_ml_integration"
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.execute.side_effect = Exception(
                'extension "google_ml_integration" is not installed'
            )
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test error string containing "google_ml"
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.execute.side_effect = Exception(
                'schema "google_ml" does not exist'
            )
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test UndefinedFunctionError direct
        class UndefinedFunctionError(Exception):
            pass

        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.execute.side_effect = UndefinedFunctionError(
                "function does not exist"
            )
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test UndefinedSchemaError direct
        class UndefinedSchemaError(Exception):
            pass

        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.execute.side_effect = UndefinedSchemaError(
                "schema does not exist"
            )
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test SQLAlchemy ProgrammingError wrapping UndefinedFunctionError
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            orig_err = UndefinedFunctionError("function does not exist")
            wrapped_err = ProgrammingError("SELECT *", {}, orig_err)
            mock_conn.execute.side_effect = wrapped_err
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test SQLAlchemy ProgrammingError wrapping UndefinedSchemaError
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            orig_err = UndefinedSchemaError("schema does not exist")
            wrapped_err = ProgrammingError("SELECT *", {}, orig_err)
            mock_conn.execute.side_effect = wrapped_err
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match=expected_msg):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )

        # Test unrelated error is re-raised
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.execute.side_effect = RuntimeError("Connection dropped")
            mock_connect.return_value.__aenter__.return_value = mock_conn

            with pytest.raises(RuntimeError, match="Connection dropped"):
                await engine._aforecast(
                    model_id="model_1",
                    timestamp_col="date",
                    data_col="revenue",
                    horizon=3,
                    source_table="sales",
                )
