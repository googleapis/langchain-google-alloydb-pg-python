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
from typing import Optional, Sequence
import uuid
import pytest
import asyncio

import pytest_asyncio
from sqlalchemy import RowMapping, text
from llama_index.core.schema import Document, TextNode, NodeRelationship

from langchain_google_alloydb_pg import AlloyDBEngine

async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()

async def afetch(engine: AlloyDBEngine, query: str) -> Sequence[RowMapping]:
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch

def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncExecute:
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

    @pytest_asyncio.fixture(scope="function")
    async def async_engine(self, db_project, db_region, db_cluster, db_instance, db_name, user, password):
        async_engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
        )

        yield async_engine

        async with async_engine._pool.connect() as conn:
          await conn.execute(text("select 3"))
          await conn.commit()

        await async_engine.close()

    # @pytest_asyncio.fixture(scope="class") ## TODO: passes only if scope is function
    # async def setup(self, async_engine):
    #     await afetch(async_engine, "select 1")
    #     yield 1
    #     await afetch(async_engine, "select 3") # Failure causes this to error out

    async def test_async_execute(self, async_engine):
        await afetch(async_engine, "select 1")
        pass
