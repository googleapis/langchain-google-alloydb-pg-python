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
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBChatMessageHistory, AlloyDBEngine

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "message_store" + str(uuid.uuid4())
table_name_async = "message_store" + str(uuid.uuid4())
user = os.environ["DB_USER"]
password = os.environ["DB_PASSWORD"]


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


@pytest_asyncio.fixture
async def engine():
    engine = AlloyDBEngine.from_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )
    engine.init_chat_history_table(table_name=table_name)
    yield engine
    # use default table for AlloyDBChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{table_name}"'
    await aexecute(engine, query)
    await engine.close()


@pytest_asyncio.fixture
async def async_engine():
    async_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )
    await async_engine.ainit_chat_history_table(table_name=table_name_async)
    yield async_engine
    # use default table for AlloyDBChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{table_name_async}"'
    await aexecute(async_engine, query)
    await async_engine.close()


def test_chat_message_history(engine: AlloyDBEngine) -> None:
    history = AlloyDBChatMessageHistory.create_sync(
        engine=engine, session_id="test", table_name=table_name
    )
    history.add_user_message("hi!")
    history.add_ai_message("whats up?")
    messages = history.messages

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    history.clear()
    assert len(history.messages) == 0


def test_chat_table(engine: Any) -> None:
    with pytest.raises(ValueError):
        AlloyDBChatMessageHistory.create_sync(
            engine=engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_chat_schema(engine: Any) -> None:
    doc_table_name = "test_table" + str(uuid.uuid4())
    engine.init_document_table(table_name=doc_table_name)
    with pytest.raises(IndexError):
        AlloyDBChatMessageHistory.create_sync(
            engine=engine, session_id="test", table_name=doc_table_name
        )

    query = f'DROP TABLE IF EXISTS "{doc_table_name}"'
    await aexecute(engine, query)


@pytest.mark.asyncio
async def test_chat_message_history_async(
    async_engine: AlloyDBEngine,
) -> None:
    history = await AlloyDBChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history.aadd_message(msg1)
    await history.aadd_message(msg2)
    messages = history.messages

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    await history.aclear()
    assert len(history.messages) == 0


@pytest.mark.asyncio
async def test_chat_message_history_sync_messages(
    async_engine: AlloyDBEngine,
) -> None:
    history1 = await AlloyDBChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    history2 = await AlloyDBChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history1.aadd_message(msg1)
    await history2.aadd_message(msg2)

    assert len(history1.messages) == 2
    assert len(history2.messages) == 2

    # verify clear() clears message history
    await history2.aclear()
    assert len(history2.messages) == 0


@pytest.mark.asyncio
async def test_chat_message_history_set_messages(
    async_engine: AlloyDBEngine,
) -> None:
    history = await AlloyDBChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="bye -_-")
    # verify setting messages property adds to message history
    history.messages = [msg1, msg2]
    assert len(history.messages) == 2


@pytest.mark.asyncio
async def test_chat_table_async(async_engine):
    with pytest.raises(ValueError):
        await AlloyDBChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_chat_schema_async(async_engine):
    table_name = "test_table" + str(uuid.uuid4())
    await async_engine.ainit_document_table(table_name=table_name)
    with pytest.raises(IndexError):
        await AlloyDBChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name
        )

    query = f'DROP TABLE IF EXISTS "{table_name}"'
    await aexecute(async_engine, query)


@pytest.mark.asyncio
async def test_cross_env_chat_message_history(engine):
    history = AlloyDBChatMessageHistory.create_sync(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    await history.aadd_message(HumanMessage(content="hi!"))
    messages = history.messages
    assert messages[0].content == "hi!"
    history.clear()

    history = await AlloyDBChatMessageHistory.create(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    history.add_message(HumanMessage(content="hi!"))
    messages = history.messages
    assert messages[0].content == "hi!"
    history.clear()


@pytest.mark.asyncio
async def test_from_engine_args_url():
    host = os.environ["IP_ADDRESS"]
    port = "5432"
    url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
    engine = AlloyDBEngine.from_engine_args(url)
    table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
    await engine.ainit_chat_history_table(table_name)

    history = AlloyDBChatMessageHistory.create_sync(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    await history.aadd_message(HumanMessage(content="hi!"))
    history.add_message(HumanMessage(content="bye!"))
    assert len(history.messages) == 2
    await history.aclear()

    history2 = await AlloyDBChatMessageHistory.create(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    await history2.aadd_message(HumanMessage(content="hi!"))
    history2.add_message(HumanMessage(content="bye!"))
    assert len(history2.messages) == 2
    history2.clear()

    await aexecute(engine, f"DROP TABLE {table_name}")
    await engine.close()


@pytest.fixture(scope="module")
def omni_host() -> str:
    return get_env_var("OMNI_HOST", "AlloyDB Omni host address")


@pytest.fixture(scope="module")
def omni_user() -> str:
    return get_env_var("OMNI_USER", "AlloyDB Omni user name")


@pytest.fixture(scope="module")
def omni_password() -> str:
    return get_env_var("OMNI_PASSWORD", "AlloyDB Omni password")


@pytest.fixture(scope="module")
def omni_database_name() -> str:
    return get_env_var("OMNI_DATABASE_ID", "AlloyDB Omni database name")


@pytest.mark.asyncio
async def omni_engine(omni_host, omni_user, omni_password, omni_database_name):
    connstring = f"postgresql+asyncpg://{omni_user}:{omni_password}@{omni_host}:5432/{omni_database_name}"
    print(f"Connecting to AlloyDB Omni with {connstring}")
    table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
    async_engine = sqlalchemy.ext.asyncio.create_async_engine(connstring)
    engine = AlloyDBEngine.from_engine(async_engine)

    history = AlloyDBChatMessageHistory.create_sync(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    await history.aadd_message(HumanMessage(content="hi!"))
    history.add_message(HumanMessage(content="bye!"))
    assert len(history.messages) == 2
    await history.aclear()

    history2 = await AlloyDBChatMessageHistory.create(
        engine=engine, session_id="test_cross", table_name=table_name
    )
    await history2.aadd_message(HumanMessage(content="hi!"))
    history2.add_message(HumanMessage(content="bye!"))
    assert len(history2.messages) == 2
    history2.clear()

    await aexecute(engine, f"DROP TABLE {table_name}")

    await engine.close()
