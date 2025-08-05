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
import sys
import uuid

import pytest
import pytest_asyncio
import sqlalchemy
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
    IVFIndex,
    ScaNNIndex,
)

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_ASYNC = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_OMNI = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_INDEX_NAME = DEFAULT_TABLE + DEFAULT_INDEX_NAME_SUFFIX
DEFAULT_INDEX_NAME_ASYNC = DEFAULT_TABLE_ASYNC + DEFAULT_INDEX_NAME_SUFFIX
DEFAULT_INDEX_NAME_OMNI = DEFAULT_TABLE_OMNI + DEFAULT_INDEX_NAME_SUFFIX
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


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


@pytest.mark.asyncio(loop_scope="class")
class TestIndex:
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

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = AlloyDBVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )

        vs.add_texts(texts, ids=ids)
        vs.drop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        vs.apply_vector_index(index)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index()

    async def test_areindex(self, vs):
        if not vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex()
            vs.apply_vector_index(index)
        vs.reindex()
        vs.reindex(DEFAULT_INDEX_NAME)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs):
        vs.drop_vector_index()
        result = vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        vs.apply_vector_index(index, concurrently=True)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        vs.apply_vector_index(index)
        assert vs.is_valid_index("secondindex")
        vs.drop_vector_index("secondindex")
        vs.drop_vector_index()

    async def test_is_valid_index(self, vs):
        is_valid = vs.is_valid_index("invalid_index")
        assert is_valid == False


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncIndex:
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

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_ASYNC}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE_ASYNC, VECTOR_SIZE)
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_ASYNC,
        )

        await vs.aadd_texts(texts, ids=ids)
        await vs.adrop_vector_index()
        yield vs

    @pytest.fixture(scope="module")
    def omni_host(self) -> str:
        return get_env_var("OMNI_HOST", "AlloyDB Omni host address")

    @pytest.fixture(scope="module")
    def omni_user(self) -> str:
        return get_env_var("OMNI_USER", "AlloyDB Omni user name")

    @pytest.fixture(scope="module")
    def omni_password(self) -> str:
        return get_env_var("OMNI_PASSWORD", "AlloyDB Omni password")

    @pytest.fixture(scope="module")
    def omni_database_name(self) -> str:
        return get_env_var("OMNI_DATABASE_ID", "AlloyDB Omni database name")

    @pytest_asyncio.fixture(scope="class")
    async def omni_engine(
        self, omni_host, omni_user, omni_password, omni_database_name
    ):
        connstring = f"postgresql+asyncpg://{omni_user}:{omni_password}@{omni_host}:5432/{omni_database_name}"
        print(f"Connecting to AlloyDB Omni with {connstring}")

        async_engine = sqlalchemy.ext.asyncio.create_async_engine(connstring)
        omni_engine = AlloyDBEngine.from_engine(async_engine)
        yield omni_engine
        await aexecute(omni_engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_OMNI}")
        await omni_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def omni_vs(self, omni_engine):
        await omni_engine.ainit_vectorstore_table(DEFAULT_TABLE_OMNI, VECTOR_SIZE)
        vs = await AlloyDBVectorStore.create(
            omni_engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_OMNI,
        )
        yield vs

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_areindex(self, vs):
        if not await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC):
            index = HNSWIndex()
            await vs.aapply_vector_index(index)
        await vs.areindex()
        await vs.areindex(DEFAULT_INDEX_NAME_ASYNC)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_dropindex(self, vs):
        await vs.adrop_vector_index()
        result = await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_is_valid_index(self, vs):
        is_valid = await vs.ais_valid_index("invalid_index")
        assert is_valid == False

    async def test_aapply_vector_index_ivf(self, vs):
        index = IVFIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        index = IVFIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_aapply_alloydb_scann_index_ScaNN(self, omni_vs):
        index = ScaNNIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await omni_vs.aset_maintenance_work_mem(index.num_leaves, VECTOR_SIZE)
        await omni_vs.aapply_vector_index(index, concurrently=True)
        assert await omni_vs.ais_valid_index(DEFAULT_INDEX_NAME_OMNI)
        index = ScaNNIndex(
            name="secondindex", distance_strategy=DistanceStrategy.COSINE_DISTANCE
        )
        await omni_vs.aapply_vector_index(index)
        assert await omni_vs.ais_valid_index("secondindex")
        await omni_vs.adrop_vector_index("secondindex")
        await omni_vs.adrop_vector_index()
