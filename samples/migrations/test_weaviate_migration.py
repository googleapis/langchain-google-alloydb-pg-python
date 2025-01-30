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
from typing import Sequence

import pytest
import pytest_asyncio
import weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore  # type: ignore
from migrate_weaviate_vectorstore_to_alloydb import main
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping
from weaviate.auth import Auth

from langchain_google_alloydb_pg import AlloyDBEngine

DEFAULT_TABLE = "test_weaviate_migration" + str(uuid.uuid4())


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


async def create_weaviate_index(
    weaviate_api_key: str, weaviate_cluster_url: str, weaviate_collection_name: str
) -> None:
    uuids = [f"{str(uuid.uuid4())}" for i in range(1000)]
    documents = [
        Document(page_content=f"content#{i}", metadata={"idv": f"{i}"})
        for i in range(1000)
    ]
    # For a locally running weaviate instance, use `weaviate.connect_to_local()`
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_cluster_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    ) as weaviate_client:
        # delete collection if exists
        try:
            weaviate_client.collections.delete(weaviate_collection_name)
        except Exception:
            pass

        db = WeaviateVectorStore.from_documents(
            documents=documents,
            ids=uuids,
            embedding=FakeEmbeddings(size=768),
            client=weaviate_client,
            index_name=weaviate_collection_name,
        )


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
    def weaviate_api_key(self) -> str:
        return get_env_var("WEAVIATE_API_KEY", "API KEY for weaviate instance")

    @pytest.fixture(scope="module")
    def weaviate_cluster_url(self) -> str:
        return get_env_var("WEAVIATE_URL", "Cluster URL for weaviate instance")

    @pytest.fixture(scope="module")
    def weaviate_collection_name(self) -> str:
        return get_env_var(
            "WEAVIATE_COLLECTION_NAME", "collection name for weaviate instance"
        )

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
        )

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await engine.close()

    async def test_weaviate(
        self,
        engine,
        capsys,
        weaviate_api_key,
        weaviate_collection_name,
        weaviate_cluster_url,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        db_user,
        db_password,
    ):
        await create_weaviate_index(
            weaviate_api_key, weaviate_cluster_url, weaviate_collection_name
        )

        await main(
            weaviate_api_key=weaviate_api_key,
            weaviate_collection_name=weaviate_collection_name,
            weaviate_text_key="text",
            weaviate_cluster_url=weaviate_cluster_url,
            vector_size=768,
            weaviate_batch_size=50,
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            alloydb_table_name=DEFAULT_TABLE,
            db_name=db_name,
            db_user=db_user,
            db_pwd=db_password,
        )

        out, err = capsys.readouterr()

        # Assert on the script's output
        assert "Error" not in err  # Check for errors
        assert "Weaviate client initiated" in out
        assert "Langchain AlloyDB client initiated" in out
        assert "Langchain Fake Embeddings service initiated." in out
        assert "Langchain AlloyDB vectorstore table created" in out
        assert "Langchain AlloyDBVectorStore initialized" in out
        assert "Weaviate client fetched all data from collection." in out
        assert "Migration completed, inserted all the batches of data to AlloyDB" in out
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 1000
