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

import pytest
import pytest_asyncio

from sqlalchemy import text

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]

from langgraph_google_alloydb_pg import AlloyDBEngine
from langgraph_google_alloydb_pg.async_checkpoint import AsyncAlloyDBSaver

        
async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


@pytest_asyncio.fixture
async def async_engine():
    async_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )
    await async_engine._ainit_checkpoint_table()
    yield async_engine
    checkpoints_query = "DROP TABLE IF EXISTS checkpoints"
    await aexecute(async_engine, checkpoints_query)
    checkpoint_writes_query = "DROP TABLE IF EXISTS checkpoint_writes"
    await aexecute(async_engine, checkpoint_writes_query)
    await async_engine.close()

