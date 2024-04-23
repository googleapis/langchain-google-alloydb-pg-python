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
from typing import List

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from google.cloud.alloydb.connector import AsyncConnector, IPTypes
from langchain_community.embeddings import FakeEmbeddings
from sqlalchemy import VARCHAR
from sqlalchemy.ext.asyncio import create_async_engine

from langchain_google_alloydb_pg import AlloyDBEngine, Column

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = VECTOR_SIZE

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (VECTOR_SIZE - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str = "default") -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (VECTOR_SIZE - 1) + [float(0.0)]


embeddings_service = FakeEmbeddingsWithDimension()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "password for AlloyDB")

    @pytest_asyncio.fixture
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            ip_type=IPTypes.PRIVATE,
        )
        yield engine

    async def test_execute(self, engine):
        await engine._aexecute("SELECT 1")

    async def test_init_table(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        await engine._aexecute(stmt)

    async def test_fetch(self, engine):
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        await engine._aexecute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init_table_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[
                Column("page", "TEXT", False),
                Column("source", "TEXT"),
            ],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await engine._afetch(stmt)
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

        await engine._aexecute(f"DROP TABLE {CUSTOM_TABLE}")

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
        await engine._aexecute("SELECT 1")

    async def test_afrom_engine(
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
                )
                return conn

            engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
            )

            engine = await AlloyDBEngine.afrom_engine(engine)
            await engine._aexecute("SELECT 1")

    async def test_column(self, engine):
        with pytest.raises(ValueError):
            Column("test", VARCHAR)


class TestEngineSync:
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
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "password for AlloyDB")

    @pytest.fixture
    def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            cluster=db_cluster,
        )
        yield engine

    def test_execute(self, engine):
        engine._execute("SELECT 1")

    def test_init_table(self, engine):
        engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = embeddings_service.embed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        engine._execute(stmt)

    def test_fetch(self, engine):
        results = engine._fetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        engine._execute(f"DROP TABLE {DEFAULT_TABLE}")

    def test_init_table_custom(self, engine):
        engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = engine._fetch(stmt)
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

        engine._execute(f"DROP TABLE {CUSTOM_TABLE}")

    def test_password(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
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
        engine._execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_from_engine(
        self,
        db_name,
        user,
        password,
    ):
        host = os.getenv("DB_HOST")
        assert host
        conn_string = f"postgresql+asyncpg://{user}:{password}@{host}:5432/{db_name}"

        pool = create_async_engine(
            conn_string,
            connect_args={"timeout": 10},
        )

        async with pool.connect() as conn:
            await conn.execute("SELECT 1")

        engine = AlloyDBEngine.from_engine(pool)
        assert engine
        engine._execute("SELECT 1")
