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
from alloydb_snippets import (
    aget_client,
    aget_vector_store,
    ainit_vector_store,
    ainsert_data,
    get_embeddings_service,
)
from pinecone_snippets import get_all_data, get_all_ids, get_client, get_index
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine, Column

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]
project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
db_user = os.environ["DB_USER"]
db_pwd = os.environ["DB_PASSWORD"]
table_name = "test-snippets-table" + str(uuid.uuid4())


async def aexecute(
    engine: AlloyDBEngine,
    query: str,
) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


@pytest.mark.asyncio(loop_scope="class")
class TestSnippetsAsync:
    async def test_pinecone(self, capsys):
        # Get all data from Pinecone index
        client = get_client(pinecone_api_key=pinecone_api_key)

        index = get_index(
            client=client,
            index_name=pinecone_index_name,
        )

        ids = get_all_ids(
            index=index,
        )
        fetched_ids, contents, embeddings, metadatas = get_all_data(
            index=index, ids=ids
        )

        # Insert all data to Langchain Alloy DB Vector Store
        alloydb_engine = await aget_client(
            project_id=project_id,
            region=region,
            cluster=cluster_id,
            instance=instance_id,
            database=db_name,
            user=db_user,
            password=db_pwd,
        )
        embeddings_service = get_embeddings_service(1024)
        await ainit_vector_store(
            engine=alloydb_engine,
            table_name=table_name,
            vector_size=1024,
            id_column=Column("langchain_id", "text", nullable=False),
        )

        vs = await aget_vector_store(
            engine=alloydb_engine,
            table_name=table_name,
            embeddings_service=embeddings_service,
        )

        inserted_ids = await ainsert_data(
            vs, contents, embeddings, metadatas, fetched_ids
        )

        out, err = capsys.readouterr()

        assert "Pinecone client initiated" in out
        assert "Pinecone index reference initiated" in out
        assert "Pinecone client fetched all ids from index" in out
        assert "Pinecone client fetched all data from index" in out

        assert "Langchain AlloyDB client initiated" in out
        assert "Langchain FakeEmbeddings service initiated" in out
        assert "Langchain AlloyDB vector store table initialized" in out
        assert "Langchain AlloyDB vector store instantiated" in out
        assert "AlloyDB client fetched all data from index" in out

        assert fetched_ids == inserted_ids
        assert len(fetched_ids) > 0

        await aexecute(alloydb_engine, f'DROP TABLE IF EXISTS "{table_name}"')
