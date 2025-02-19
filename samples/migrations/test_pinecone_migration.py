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
import time
import uuid
from typing import Sequence

import pytest
import pytest_asyncio
from google.cloud.alloydb.connector import IPTypes
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_pinecone import PineconeVectorStore  # type: ignore
from migrate_pinecone_vectorstore_to_alloydb import main
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_alloydb_pg import AlloyDBEngine

DEFAULT_TABLE = "test_pinecone_migration" + str(uuid.uuid4())


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


def create_pinecone_index(
    pinecone_index_name: str, pinecone_api_key: str, project_id: str
) -> None:
    client = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in client.list_indexes()]
    if pinecone_index_name in existing_indexes:
        # Assume documents already added if index exists
        return
    # Create the index and add test documents
    client.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not client.describe_index(pinecone_index_name).status["ready"]:
        time.sleep(1)

    index = client.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=FakeEmbeddings(size=768),
    )

    ids = ids = [f"{str(uuid.uuid4())}" for i in range(1000)]
    documents = [
        Document(page_content=f"content#{i}", metadata={"idv": f"{i}"})
        for i in range(1000)
    ]
    vector_store.add_documents(documents=documents, ids=ids)


@pytest.mark.asyncio(loop_scope="class")
class TestMigrations:
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
    def db_user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def db_password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest.fixture(scope="module")
    def pinecone_api_key(self) -> str:
        return get_env_var("PINECONE_API_KEY", "API KEY for pinecone instance")

    @pytest.fixture(scope="module")
    def pinecone_index_name(self) -> str:
        return get_env_var("PINECONE_INDEX_NAME", "index name for pinecone instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        db_user,
        db_password,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=db_user,
            password=db_password,
            ip_type=IPTypes.PUBLIC,
        )

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await engine.close()

    async def test_pinecone(
        self,
        engine,
        capsys,
        pinecone_api_key,
        pinecone_index_name,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        db_user,
        db_password,
    ):
        create_pinecone_index(pinecone_index_name, pinecone_api_key, db_project)

        await main(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            pinecone_namespace="",
            vector_size=768,
            pinecone_batch_size=50,
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            alloydb_table=DEFAULT_TABLE,
            db_name=db_name,
            db_user=db_user,
            db_pwd=db_password,
        )

        out, err = capsys.readouterr()

        # Assert on the script's output
        assert "Error" not in err  # Check for errors
        assert "Pinecone index reference initiated" in out
        assert "Langchain AlloyDB client initiated" in out
        assert "Langchain AlloyDB vectorstore table created" in out
        assert "Langchain Fake Embeddings service initiated" in out
        assert "Langchain AlloyDBVectorStore initialized" in out
        assert "Pinecone client fetched all ids from index" in out
        assert "Pinecone client fetched all data from index" in out
        assert "Migration completed, inserted all the batches of data to AlloyDB" in out
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 1000
