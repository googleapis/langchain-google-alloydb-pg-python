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


import asyncio
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBVectorStore,
    HybridSearchConfig,
)
from langchain_google_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
    IVFIndex,
    ScaNNIndex,
)

UUID_STR = str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE = "table" + UUID_STR
DEFAULT_HYBRID_TABLE = "hybrid" + UUID_STR
DEFAULT_INDEX_NAME = DEFAULT_INDEX_NAME_SUFFIX + UUID_STR
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


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


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
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_HYBRID_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )

        await vs.aadd_texts(texts, ids=ids)
        await vs.adrop_vector_index()
        yield vs

    async def test_aapply_vector_index_ivf(self, vs):
        index = IVFIndex(
            name=DEFAULT_INDEX_NAME,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index()

    async def test_areindex(self, vs):
        if not await vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex()
            await vs.aapply_vector_index(index)
        await vs.areindex(DEFAULT_INDEX_NAME)
        await vs.areindex(DEFAULT_INDEX_NAME)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs):
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)
        result = await vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(
            name=DEFAULT_INDEX_NAME, distance_strategy=DistanceStrategy.EUCLIDEAN
        )
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_is_valid_index(self, vs):
        is_valid = await vs.is_valid_index("invalid_index")
        assert is_valid == False

    async def test_apply_default_name_vector_index(self, vs):
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert index.name is None
        assert await vs.is_valid_index()
        await vs.adrop_vector_index()

    async def test_aapply_vector_index_non_hybrid_search_vs(self, vs):
        with pytest.raises(ValueError):
            await vs.aapply_hybrid_search_index()

    async def test_aapply_hybrid_search_index_table_without_tsv_column(
        self, engine, vs
    ):
        # overwriting vs to get a hybrid vs
        tsv_index_name = "index_without_tsv_column_" + UUID_STR
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            hybrid_search_config=HybridSearchConfig(index_name=tsv_index_name),
        )
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.aapply_hybrid_search_index()
        assert await vs.is_valid_index(tsv_index_name)
        await vs.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False

    async def test_aapply_hybrid_search_index_table_with_tsv_column(self, engine):
        tsv_index_name = "index_without_tsv_column_" + UUID_STR
        config = HybridSearchConfig(
            tsv_column="tsv_column",
            tsv_lang="pg_catalog.english",
            index_name=tsv_index_name,
        )
        await engine._ainit_vectorstore_table(
            DEFAULT_HYBRID_TABLE,
            VECTOR_SIZE,
            hybrid_search_config=config,
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_HYBRID_TABLE,
            hybrid_search_config=config,
        )
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.aapply_hybrid_search_index()
        assert await vs.is_valid_index(tsv_index_name)
        await vs.areindex(tsv_index_name)
        assert await vs.is_valid_index(tsv_index_name)
        await vs.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False


@pytest.mark.asyncio
class TestAsyncVectorStoreIndexUnit:
    @pytest.fixture
    def vs_and_conn(self):
        vs = AsyncAlloyDBVectorStore.__new__(AsyncAlloyDBVectorStore)
        vs.table_name = "test_table"
        vs.schema_name = "public"
        vs.embedding_column = "embedding"

        mock_conn = AsyncMock()
        mock_conn.execution_options = AsyncMock(return_value=mock_conn)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_conn
        mock_ctx.__aexit__.return_value = None

        mock_pool = MagicMock()
        mock_pool.begin.return_value = mock_ctx
        mock_pool.connect.return_value = mock_ctx

        vs.engine = MagicMock(_pool=mock_pool)
        return vs, mock_conn

    async def test_aapply_vector_index_leaves_index_name_none(self, vs_and_conn):
        vs, mock_conn = vs_and_conn
        index = HNSWIndex(name=None)
        assert index.name is None

        await vs.aapply_vector_index(index)

        # Calling aapply_vector_index with index.name = None must leave index.name as None
        assert index.name is None
        assert mock_conn.execute.call_count == 1
        executed_query = str(mock_conn.execute.call_args[0][0])
        expected_index_name = f"test_table{DEFAULT_INDEX_NAME_SUFFIX}"
        assert f'"{expected_index_name}"' in executed_query

    async def test_aapply_vector_index_preserves_custom_name(self, vs_and_conn):
        vs, mock_conn = vs_and_conn
        index = HNSWIndex(name="my_custom_index")

        await vs.aapply_vector_index(index)

        assert index.name == "my_custom_index"
        assert mock_conn.execute.call_count == 1
        executed_query = str(mock_conn.execute.call_args[0][0])
        assert '"my_custom_index"' in executed_query

    async def test_aapply_vector_index_with_explicit_name_arg(self, vs_and_conn):
        vs, mock_conn = vs_and_conn
        index = HNSWIndex(name=None)

        await vs.aapply_vector_index(index, name="explicit_index_name")

        assert index.name is None
        assert mock_conn.execute.call_count == 1
        executed_query = str(mock_conn.execute.call_args[0][0])
        assert '"explicit_index_name"' in executed_query

    async def test_aapply_vector_index_concurrently_leaves_index_name_none(
        self, vs_and_conn
    ):
        vs, mock_conn = vs_and_conn
        index = HNSWIndex(name=None)

        await vs.aapply_vector_index(index, concurrently=True)

        assert index.name is None
        executed_query = str(mock_conn.execute.call_args[0][0])
        expected_index_name = f"test_table{DEFAULT_INDEX_NAME_SUFFIX}"
        assert f'"{expected_index_name}"' in executed_query
        assert "CONCURRENTLY" in executed_query

    async def test_aapply_vector_index_scann_leaves_index_name_none(self, vs_and_conn):
        vs, mock_conn = vs_and_conn
        index = ScaNNIndex(name=None)
        assert index.name is None

        await vs.aapply_vector_index(index)

        assert index.name is None
        executed_calls = [str(call[0][0]) for call in mock_conn.execute.call_args_list]
        expected_index_name = f"test_table{DEFAULT_INDEX_NAME_SUFFIX}"
        assert any(f'"{expected_index_name}"' in call for call in executed_calls)
        assert any("SET LOCAL maintenance_work_mem" in call for call in executed_calls)

    async def test_aapply_vector_index_ivfflat_leaves_index_name_none(
        self, vs_and_conn
    ):
        vs, mock_conn = vs_and_conn
        index = IVFFlatIndex(name=None)
        assert index.name is None

        await vs.aapply_vector_index(index)

        assert index.name is None
        executed_query = str(mock_conn.execute.call_args[0][0])
        expected_index_name = f"test_table{DEFAULT_INDEX_NAME_SUFFIX}"
        assert f'"{expected_index_name}"' in executed_query

    async def test_aapply_vector_index_reuse_across_tables(self):
        mock_conn1 = AsyncMock()
        mock_ctx1 = AsyncMock()
        mock_ctx1.__aenter__.return_value = mock_conn1
        mock_ctx1.__aexit__.return_value = None
        mock_pool1 = MagicMock()
        mock_pool1.begin.return_value = mock_ctx1

        mock_conn2 = AsyncMock()
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__.return_value = mock_conn2
        mock_ctx2.__aexit__.return_value = None
        mock_pool2 = MagicMock()
        mock_pool2.begin.return_value = mock_ctx2

        vs1 = AsyncAlloyDBVectorStore.__new__(AsyncAlloyDBVectorStore)
        vs1.table_name = "table_one"
        vs1.schema_name = "public"
        vs1.embedding_column = "embedding"
        vs1.engine = MagicMock(_pool=mock_pool1)

        vs2 = AsyncAlloyDBVectorStore.__new__(AsyncAlloyDBVectorStore)
        vs2.table_name = "table_two"
        vs2.schema_name = "public"
        vs2.embedding_column = "embedding"
        vs2.engine = MagicMock(_pool=mock_pool2)

        shared_index = HNSWIndex(name=None)

        await vs1.aapply_vector_index(shared_index)
        assert shared_index.name is None
        query1 = str(mock_conn1.execute.call_args[0][0])
        assert f'"table_one{DEFAULT_INDEX_NAME_SUFFIX}"' in query1

        await vs2.aapply_vector_index(shared_index)
        assert shared_index.name is None
        query2 = str(mock_conn2.execute.call_args[0][0])
        assert f'"table_two{DEFAULT_INDEX_NAME_SUFFIX}"' in query2


class TestSyncVectorStoreIndexUnit:
    def test_sync_apply_vector_index_leaves_index_name_none(self):
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_conn
        mock_ctx.__aexit__.return_value = None
        mock_pool.begin.return_value = mock_ctx

        async_vs = AsyncAlloyDBVectorStore.__new__(AsyncAlloyDBVectorStore)
        async_vs.table_name = "sync_table"
        async_vs.schema_name = "public"
        async_vs.embedding_column = "embedding"
        async_vs.engine = MagicMock(_pool=mock_pool)

        sync_vs = AlloyDBVectorStore.__new__(AlloyDBVectorStore)
        mock_engine = MagicMock()
        mock_engine._run_as_sync.side_effect = lambda coro: asyncio.run(coro)
        sync_vs._engine = mock_engine
        sync_vs._PGVectorStore__vs = async_vs

        index = HNSWIndex(name=None)
        sync_vs.apply_vector_index(index)

        assert index.name is None
        assert mock_conn.execute.call_count == 1
        executed_query = str(mock_conn.execute.call_args[0][0])
        expected_index_name = f"sync_table{DEFAULT_INDEX_NAME_SUFFIX}"
        assert f'"{expected_index_name}"' in executed_query
