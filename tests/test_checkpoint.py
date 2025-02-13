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
from typing import Any, Sequence, Tuple

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_alloydb_pg.checkpoint import AlloyDBSaver
from langchain_google_alloydb_pg.engine import AlloyDBEngine

write_config: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config: RunnableConfig = {"configurable": {"thread_id": "1"}}

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "checkpoint" + str(uuid.uuid4())
table_name_writes = table_name + "_writes"
table_name_async = "checkpoint" + str(uuid.uuid4())
table_name_writes_async = table_name_async + "_writes"

checkpoint: Checkpoint = {
    "v": 1,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {"my_key": "meow", "node": "node"},
    "channel_versions": {
        "__start__": 2,
        "my_key": 3,
        "start:node": 3,
        "node": 3,
    },
    "versions_seen": {
        "__input__": {},
        "__start__": {"__start__": 1},
        "node": {"start:node": 2},
    },
    "pending_sends": [],
}


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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


@pytest_asyncio.fixture
async def engine():
    engine = AlloyDBEngine.from_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )
    engine.init_checkpoint_table(table_name=table_name)
    yield engine
    # use default table
    await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')
    await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name_writes}"')
    await engine.close()
    await engine._connector.close()


@pytest_asyncio.fixture  ##(scope="module")
async def async_engine():
    async_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )

    await async_engine.ainit_checkpoint_table(table_name=table_name_async)

    yield async_engine

    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name_async}"')
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name_writes_async}"')
    await async_engine.close()
    await async_engine._connector.close()


@pytest_asyncio.fixture  ##(scope="module")
def checkpointer(engine):
    checkpointer = AlloyDBSaver.create_sync(engine, table_name)
    yield checkpointer


async def test_checkpoint(
    engine: AlloyDBEngine,
    checkpointer: AlloyDBSaver,
) -> None:
    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }
    # Verify if updated configuration after storing the checkpoint is correct
    next_config = checkpointer.put(write_config, checkpoint, {}, {})
    assert dict(next_config) == test_config

    results = await afetch(engine, f'SELECT * FROM "{table_name}"')
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(engine, f'TRUNCATE TABLE "{table_name}"')


def test_checkpoint_table(engine: Any) -> None:
    with pytest.raises(ValueError):
        AlloyDBSaver.create_sync(engine=engine, table_name="doesnotexist")


@pytest.mark.asyncio
async def test_checkpoint_async(
    async_engine: AlloyDBEngine,
    checkpointer: AlloyDBSaver,
) -> None:

    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }
    # Verify if updated configuration after storing the checkpoint is correct
    next_config = await checkpointer.aput(write_config, checkpoint, {}, {})
    assert dict(next_config) == test_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(async_engine, f'SELECT * FROM "{table_name}"')
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(async_engine, f'TRUNCATE TABLE "{table_name}"')


@pytest.mark.asyncio
async def test_chat_table_async(async_engine):
    with pytest.raises(ValueError):
        await AlloyDBSaver.create(engine=async_engine, table_name="doesnotexist")
