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
from langchain_core.embeddings import DeterministicFakeEmbedding
from metadata_filtering_data import FILTERING_TEST_CASES, METADATAS
from PIL import Image
from sqlalchemy import text

from langchain_google_alloydb_pg import AlloyDBEngine, Column
from langchain_google_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import (
    DistanceStrategy,
    HNSWQueryOptions,
    ScaNNQueryOptions,
)

CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
IMAGE_TABLE = "test_image_table" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768
sync_method_exception_str = "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

# Note: The following texts are chosen to produce diverse
# similarity scores when using the DeterministicFakeEmbedding service. This ensures
# that the test cases can effectively validate the filtering and scoring logic.
# The scoring might be different if using a different embedding service.
texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


class FakeImageEmbedding(DeterministicFakeEmbedding):

    def embed_image(self, image_paths: list[str]) -> list[list[float]]:
        return [self.embed_query(path) for path in image_paths]


image_embedding_service = FakeImageEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(
    engine: AlloyDBEngine,
    query: str,
) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStoreSearch:
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
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {IMAGE_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine):
        await engine._ainit_vectorstore_table(
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

        vs_custom = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        await vs_custom.aadd_documents(docs, ids=ids)
        yield vs_custom

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom_scann_query_option(self, engine, vs_custom):
        vs_custom_scann_query_option = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=ScaNNQueryOptions(
                num_leaves_to_search=1, pre_reordering_num_neighbors=2
            ),
        )
        yield vs_custom_scann_query_option

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self):
        red_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_red.jpg"
        green_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_green.jpg"
        blue_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_blue.jpg"
        gcs_uri = "gs://github-repo/img/vision/google-cloud-next.jpeg"
        image = Image.new("RGB", (100, 100), color="red")
        image.save(red_uri)
        image = Image.new("RGB", (100, 100), color="green")
        image.save(green_uri)
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(blue_uri)
        image_uris = [red_uri, green_uri, blue_uri, gcs_uri]
        yield image_uris
        for uri in image_uris:
            try:
                os.remove(uri)
            except FileNotFoundError:
                pass

    @pytest_asyncio.fixture(scope="class")
    async def image_vs(self, engine, image_uris):
        await engine._ainit_vectorstore_table(IMAGE_TABLE, VECTOR_SIZE)
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=image_embedding_service,
            table_name=IMAGE_TABLE,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        )
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        await vs.aadd_images(image_uris, ids=ids)
        yield vs

    async def test_asimilarity_search_scann(self, vs_custom_scann_query_option):
        results = await vs_custom_scann_query_option.asimilarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]
        results = await vs_custom_scann_query_option.asimilarity_search(
            "foo", k=1, filter={'mycontent': 'bar'}
        )
        assert results == [Document(page_content="bar", id=ids[1])]

    async def test_asimilarity_search_image(self, image_vs, image_uris):
        results = await image_vs.asimilarity_search_image(image_uris[0], k=1)
        assert len(results) == 1
        assert results[0].metadata["image_uri"] == image_uris[0]
        results = await image_vs.asimilarity_search_image(image_uris[3], k=1)
        assert len(results) == 1
        assert results[0].metadata["image_uri"] == image_uris[3]

    async def test_similarity_search_image(self, image_vs, image_uris):
        with pytest.raises(NotImplementedError):
            await image_vs.similarity_search_image(image_uris[0], k=1)
