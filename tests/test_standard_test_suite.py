# Copyright 2025 Google LLC
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

import pytest
import pytest_asyncio
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column

DEFAULT_TABLE = "test_table_standard_test_suite" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync_standard_test_suite" + str(uuid.uuid4())


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


@pytest.mark.filterwarnings("ignore")
@pytest.mark.asyncio
class TestStandardSuiteSync(VectorStoreIntegrationTests):
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
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest_asyncio.fixture(loop_scope="function")
    async def sync_engine(
        self, db_project, db_region, db_cluster, db_instance, db_name
    ):
        sync_engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield sync_engine
        await aexecute(sync_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')
        await sync_engine.close()

    @pytest.fixture(scope="function")
    def vectorstore(self, sync_engine):
        """Get an empty vectorstore for unit tests."""
        sync_engine.init_vectorstore_table(
            DEFAULT_TABLE_SYNC,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = AlloyDBVectorStore.create_sync(
            sync_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs


@pytest.mark.filterwarnings("ignore")
@pytest.mark.asyncio
class TestStandardSuiteAsync(VectorStoreIntegrationTests):
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
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest_asyncio.fixture(loop_scope="function")
    async def async_engine(
        self, db_project, db_region, db_cluster, db_instance, db_name
    ):
        async_engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield async_engine
        await aexecute(async_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await async_engine.close()

    @pytest_asyncio.fixture(loop_scope="function")
    async def vectorstore(self, async_engine):
        """Get an empty vectorstore for unit tests."""
        await async_engine.ainit_vectorstore_table(
            DEFAULT_TABLE,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = await AlloyDBVectorStore.create(
            async_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE,
        )

        yield vs
