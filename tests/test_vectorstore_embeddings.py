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
from langchain_core.documents import Document
from sqlalchemy import text

from langchain_google_alloydb_pg import (
    AlloyDBEmbeddings,
    AlloyDBEngine,
    AlloyDBModelManager,
    AlloyDBVectorStore,
    Column,
)
from langchain_google_alloydb_pg.indexes import DistanceStrategy, HNSWQueryOptions

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
VECTOR_SIZE = 768


texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]


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
class TestVectorStoreEmbeddings:
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
    async def engine(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
    ):
        engine = await AlloyDBEngine.afrom_instance(
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
    async def embeddings_service(self, engine):
        model_manager = await AlloyDBModelManager.create(engine=engine)
        model = await model_manager.aget_model(model_id=DEFAULT_EMBEDDING_MODEL)
        if not model:
            # create model if not exists
            await model_manager.acreate_model(
                model_id=DEFAULT_EMBEDDING_MODEL,
                model_provider="google",
                model_qualified_name=DEFAULT_EMBEDDING_MODEL,  # assuming model is built-in
                model_type="text_embedding",
            )
        return await AlloyDBEmbeddings.create(engine, DEFAULT_EMBEDDING_MODEL)

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine, embeddings_service):
        await engine.ainit_vectorstore_table(
            DEFAULT_TABLE, VECTOR_SIZE, store_metadata=False, overwrite_existing=True
        )
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        await vs.aadd_documents(docs, ids=ids)
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def engine_sync(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
    ):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine_sync, embeddings_service):
        engine_sync.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[
                Column("page", "TEXT"),
                Column("source", "TEXT"),
            ],
            store_metadata=False,
        )

        vs_custom = AlloyDBVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        vs_custom.add_documents(docs, ids=ids)
        yield vs_custom

    async def test_asimilarity_search(self, vs):
        results = await vs.asimilarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]
        results = await vs.asimilarity_search("foo", k=1, filter="content = 'bar'")
        assert results == [Document(page_content="bar", id=ids[1])]

    async def test_asimilarity_search_score(self, vs):
        results = await vs.asimilarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_asimilarity_search_by_vector(self, vs, embeddings_service):
        search_embedding = embeddings_service.embed_query("foo")
        results = await vs.asimilarity_search_by_vector(search_embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        results = await vs.asimilarity_search_with_score_by_vector(search_embedding)
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_similarity_search_with_relevance_scores_threshold_cosine(self, vs):
        score_threshold = {"score_threshold": 0}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 4

        score_threshold = {"score_threshold": 0.65}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.8}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

    async def test_similarity_search_with_relevance_scores_threshold_euclidean(
        self, engine, embeddings_service
    ):
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )

        score_threshold = {"score_threshold": 0.9}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

    async def test_amax_marginal_relevance_search(self, vs):
        results = await vs.amax_marginal_relevance_search("bar")
        assert results[0] == Document(page_content="bar", id=ids[1])
        results = await vs.amax_marginal_relevance_search(
            "bar", filter="content = 'boo'"
        )
        assert results[0] == Document(page_content="boo", id=ids[3])

    async def test_amax_marginal_relevance_search_vector(self, vs, embeddings_service):
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar", id=ids[1])

    async def test_amax_marginal_relevance_search_vector_score(
        self, vs, embeddings_service
    ):
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])


class TestVectorStoreEmbeddingsSync:
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
    async def engine_sync(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_SYNC}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    def embeddings_service(self, engine_sync):
        return AlloyDBEmbeddings.create_sync(engine_sync, DEFAULT_EMBEDDING_MODEL)

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine_sync, embeddings_service):
        engine_sync.init_vectorstore_table(
            DEFAULT_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[
                Column("page", "TEXT"),
                Column("source", "TEXT"),
            ],
            store_metadata=False,
        )

        vs_custom = await AlloyDBVectorStore.create(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_SYNC,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        vs_custom.add_documents(docs, ids=ids)
        yield vs_custom

    def test_similarity_search(self, vs_custom):
        results = vs_custom.similarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]
        results = vs_custom.similarity_search("foo", k=1, filter="mycontent = 'bar'")
        assert results == [Document(page_content="bar", id=ids[1])]

    def test_similarity_search_score(self, vs_custom):
        results = vs_custom.similarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    def test_similarity_search_by_vector(self, vs_custom, embeddings_service):
        embedding = embeddings_service.embed_query("foo")
        results = vs_custom.similarity_search_by_vector(embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        results = vs_custom.similarity_search_with_score_by_vector(embedding)
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    def test_max_marginal_relevance_search(self, vs_custom):
        results = vs_custom.max_marginal_relevance_search("bar")
        assert results[0] == Document(page_content="bar", id=ids[1])
        results = vs_custom.max_marginal_relevance_search(
            "bar", filter="mycontent = 'boo'"
        )
        assert results[0] == Document(page_content="boo", id=ids[3])

    def test_max_marginal_relevance_search_vector(self, vs_custom, embeddings_service):
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar", id=ids[1])

    def test_max_marginal_relevance_search_vector_score(
        self, vs_custom, embeddings_service
    ):
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])
