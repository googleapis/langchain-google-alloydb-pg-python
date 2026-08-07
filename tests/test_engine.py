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

import os
import uuid
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from google.cloud.alloydb.connector import AsyncConnector, IPTypes
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import VARCHAR, text
from sqlalchemy.engine import URL
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from langchain_google_alloydb_pg import AlloyDBEngine, Column, HybridSearchConfig

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE = "test_table_custom_int_id" + str(uuid.uuid4()).replace("-", "_")
HYBRID_SEARCH_TABLE = "hybrid" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE_SYNC = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE_SYNC = "test_table_custom_int_id" + str(uuid.uuid4()).replace(
    "-", "_"
)
HYBRID_SEARCH_TABLE_SYNC = "hybrid_sync" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
host = os.environ.get("IP_ADDRESS", "127.0.0.1")


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(
    engine: AlloyDBEngine,
    query: str,
) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


async def afetch(engine: AlloyDBEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


@pytest.mark.asyncio
class TestEngineAsync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            engine_args={
                # add some connection args to validate engine_args works correctly
                "pool_size": 3,
                "max_overflow": 2,
            },
        )
        yield engine
        await aexecute(engine, f'DROP TABLE "{CUSTOM_TABLE}"')
        await aexecute(engine, f'DROP TABLE "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE "{INT_ID_CUSTOM_TABLE}"')
        await aexecute(engine, f'DROP TABLE "{HYBRID_SEARCH_TABLE}"')
        await engine.close()

    async def test_init_table(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        # Note: DeterministicFakeEmbedding generates a numpy array, converting to list a list of float values
        embedding_string = [float(dimension) for dimension in embedding]
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding_string}');"
        await aexecute(engine, stmt)

    async def test_engine_args(self, engine):
        assert "Pool size: 3" in engine._pool.pool.status()

    async def test_init_table_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_init_table_with_int_id(self, engine):
        await engine.ainit_vectorstore_table(
            INT_ID_CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable="False"),
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{INT_ID_CUSTOM_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "integer_id", "data_type": "integer"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_password(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        AlloyDBEngine._connector = None
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            user=user,
            password=password,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        AlloyDBEngine._connector = None
        await engine.close()

    async def test_from_engine(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        async with AsyncConnector() as connector:

            async def getconn() -> asyncpg.Connection:
                conn = await connector.connect(  # type: ignore
                    f"projects/{db_project}/locations/{db_region}/clusters/{db_cluster}/instances/{db_instance}",
                    "asyncpg",
                    user=user,
                    password=password,
                    db=db_name,
                    enable_iam_auth=False,
                    ip_type=IPTypes.PUBLIC,
                )
                return conn

            engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
            )

            engine = AlloyDBEngine.from_engine(engine)
            await aexecute(engine, "SELECT 1")
            await engine.close()

    async def test_from_connection_string(
        self,
        db_name,
        user,
        password,
    ):
        port = "5432"
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        engine = AlloyDBEngine.from_connection_string(
            url,
            echo=True,
            poolclass=NullPool,
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

        engine = AlloyDBEngine.from_connection_string(
            URL.create("postgresql+asyncpg", user, password, host, port, db_name)
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_from_engine_args_url(
        self,
        db_name,
        user,
        password,
    ):
        port = "5432"
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        engine = AlloyDBEngine.from_engine_args(
            url,
            echo=True,
            poolclass=NullPool,
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

        engine = AlloyDBEngine.from_engine_args(
            URL.create("postgresql+asyncpg", user, password, host, port, db_name)
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_from_engine_args_url_error(
        self,
        db_name,
        user,
        password,
    ):
        port = "5432"
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        with pytest.raises(TypeError):
            engine = AlloyDBEngine.from_engine_args(url, random=False)
        with pytest.raises(ValueError):
            AlloyDBEngine.from_engine_args(
                f"postgresql+pg8000://{user}:{password}@{host}:{port}/{db_name}",
            )
        with pytest.raises(ValueError):
            AlloyDBEngine.from_engine_args(
                URL.create("postgresql+pg8000", user, password, host, port, db_name)
            )

    async def test_column(self, engine):
        with pytest.raises(ValueError):
            Column("test", VARCHAR)
        with pytest.raises(ValueError):
            Column(1, "INTEGER")

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_ainit_checkpoint_writes_table(self, engine):
        table_name = f"checkpoint{uuid.uuid4()}"
        table_name_writes = f"{table_name}_writes"
        await engine.ainit_checkpoint_table(table_name=table_name)
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name_writes}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "thread_id", "data_type": "text"},
            {"column_name": "checkpoint_ns", "data_type": "text"},
            {"column_name": "checkpoint_id", "data_type": "text"},
            {"column_name": "task_id", "data_type": "text"},
            {"column_name": "idx", "data_type": "integer"},
            {"column_name": "channel", "data_type": "text"},
            {"column_name": "type", "data_type": "text"},
            {"column_name": "blob", "data_type": "bytea"},
            {"column_name": "task_path", "data_type": "text"},
        ]
        for row in results:
            assert row in expected
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "thread_id", "data_type": "text"},
            {"column_name": "checkpoint_ns", "data_type": "text"},
            {"column_name": "checkpoint_id", "data_type": "text"},
            {"column_name": "parent_checkpoint_id", "data_type": "text"},
            {"column_name": "checkpoint", "data_type": "bytea"},
            {"column_name": "metadata", "data_type": "bytea"},
            {"column_name": "type", "data_type": "text"},
        ]
        for row in results:
            assert row in expected
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name_writes}"')

    async def test_init_table_hybrid_search(self, engine):
        await engine.ainit_vectorstore_table(
            HYBRID_SEARCH_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
            hybrid_search_config=HybridSearchConfig(),
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{HYBRID_SEARCH_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "my-content_tsv", "data_type": "tsvector"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected


@pytest.mark.asyncio
class TestEngineSync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f'DROP TABLE "{CUSTOM_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE "{DEFAULT_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE "{INT_ID_CUSTOM_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE "{HYBRID_SEARCH_TABLE_SYNC}"')
        await engine.close()

    async def test_init_table(self, engine):
        engine.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        # Note: DeterministicFakeEmbedding generates a numpy array, converting to list a list of float values
        embedding_string = [float(dimension) for dimension in embedding]
        stmt = f"INSERT INTO {DEFAULT_TABLE_SYNC} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding_string}');"
        await aexecute(engine, stmt)

    async def test_init_table_custom(self, engine):
        engine.init_vectorstore_table(
            CUSTOM_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_init_table_with_int_id(self, engine):
        engine.init_vectorstore_table(
            INT_ID_CUSTOM_TABLE_SYNC,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable="False"),
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{INT_ID_CUSTOM_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "integer_id", "data_type": "integer"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_password(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        AlloyDBEngine._connector = None
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            user=user,
            password=password,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        AlloyDBEngine._connector = None
        await engine.close()

    async def test_engine_constructor_key(
        self,
        engine,
    ):
        key = object()
        with pytest.raises(Exception):
            AlloyDBEngine(key, engine)

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_init_checkpoints_table(self, engine):
        table_name = f"checkpoint{uuid.uuid4()}"
        table_name_writes = f"{table_name}_writes"
        engine.init_checkpoint_table(table_name=table_name)
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "thread_id", "data_type": "text"},
            {"column_name": "checkpoint_ns", "data_type": "text"},
            {"column_name": "checkpoint_id", "data_type": "text"},
            {"column_name": "parent_checkpoint_id", "data_type": "text"},
            {"column_name": "type", "data_type": "text"},
            {"column_name": "checkpoint", "data_type": "bytea"},
            {"column_name": "metadata", "data_type": "bytea"},
        ]
        for row in results:
            assert row in expected
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name_writes}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "thread_id", "data_type": "text"},
            {"column_name": "checkpoint_ns", "data_type": "text"},
            {"column_name": "checkpoint_id", "data_type": "text"},
            {"column_name": "task_id", "data_type": "text"},
            {"column_name": "idx", "data_type": "integer"},
            {"column_name": "channel", "data_type": "text"},
            {"column_name": "type", "data_type": "text"},
            {"column_name": "blob", "data_type": "bytea"},
            {"column_name": "task_path", "data_type": "text"},
        ]
        for row in results:
            assert row in expected
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name_writes}"')

    async def test_init_table_hybrid_search(self, engine):
        engine.init_vectorstore_table(
            HYBRID_SEARCH_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
            hybrid_search_config=HybridSearchConfig(),
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{HYBRID_SEARCH_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "my-content_tsv", "data_type": "tsvector"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected


class TestEngineUnit:
    @pytest.fixture
    def engine(self):
        eng = AlloyDBEngine.__new__(AlloyDBEngine)
        eng._pool = MagicMock()
        eng._run_as_sync = MagicMock()

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
            mock_result.mappings.return_value.fetchall.return_value = [
                {"prediction": 1.0},
                {"prediction": 2.0},
            ]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            results = await engine.aforecast(
                model_id="test_model",
                source_table="test_table",
                source_query=None,
                data_col="data",
                timestamp_col="ts",
                horizon=5,
            )
            assert len(results) == 2
            assert results[0]["prediction"] == 1.0

    @pytest.mark.asyncio
    async def test_aforecast_with_optional_params(self, engine):
        """Test aforecast with source_query and conf_level."""
        with patch.object(
            engine, "_aforecast", new_callable=AsyncMock
        ) as mock_aforecast:
            mock_aforecast.return_value = [{"prediction": 42.0}]
            results = await engine.aforecast(
                model_id="test_model",
                source_table="test_table",
                source_query="SELECT * FROM data",
                data_col="data",
                timestamp_col="ts",
                horizon=10,
                conf_level=0.95,
            )
            assert len(results) == 1
            assert results[0]["prediction"] == 42.0
            mock_aforecast.assert_called_once_with(
                "test_model",
                "test_table",
                "ts",
                "data",
                10,
                "SELECT * FROM data",
                0.95,
            )

    def test_forecast(self, engine):
        """Test that forecast evaluates via _run_as_sync to proxy the google_ml.forecast."""
        engine._run_as_sync.return_value = [{"prediction": 1.0}]
        results = engine.forecast(
            model_id="test_model",
            source_table="test_table",
            source_query=None,
            data_col="data",
            timestamp_col="ts",
            horizon=5,
        )
        assert len(results) == 1
        assert results[0]["prediction"] == 1.0

    def test_forecast_with_optional_params(self, engine):
        """Test forecast with source_query and conf_level."""
        engine._run_as_sync.return_value = [{"prediction": 42.0}]
        results = engine.forecast(
            model_id="test_model",
            source_table="test_table",
            source_query="SELECT * FROM data",
            data_col="data",
            timestamp_col="ts",
            horizon=10,
            conf_level=0.95,
        )
        assert len(results) == 1
        assert results[0]["prediction"] == 42.0
        engine._run_as_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_private_aforecast(self, engine):
        """Test direct _aforecast execution and mapping parsing."""
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value.fetchall.return_value = [
                {"forecast_timestamp": "2026-08-07", "forecast_value": 100.0}
            ]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            results = await engine._aforecast(
                model_id="model_1",
                source_table="sales",
                timestamp_col="date",
                data_col="revenue",
                horizon=3,
                source_query="SELECT * FROM sales WHERE active = true",
                conf_level=0.9,
            )
            assert len(results) == 1
            assert results[0]["forecast_value"] == 100.0
            assert mock_conn.execute.called
