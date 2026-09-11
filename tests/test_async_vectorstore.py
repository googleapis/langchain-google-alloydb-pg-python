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

import json
import os
import uuid
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from PIL import Image
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_alloydb_pg import AlloyDBEngine, Column
from langchain_google_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import (
    DistanceStrategy,
    ScaNNIndex,
)

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "custom" + str(uuid.uuid4())
IMAGE_TABLE = "image" + str(uuid.uuid4())
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]
id_column_as_metadata = [{"id": str(i)} for i in range(len(texts))]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


class FakeImageEmbedding(DeterministicFakeEmbedding):

    def embed_image(self, image_paths: list[str]) -> list[list[float]]:
        return [self.embed_query(f"Image Path: {path}") for path in image_paths]


image_embedding_service = FakeImageEmbedding(size=VECTOR_SIZE)


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


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStore:
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

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        host = os.environ.get("OMNI_HOST") or os.environ.get("IP_ADDRESS")
        user = os.environ.get("OMNI_USER") or os.environ.get("DB_USER", "postgres")
        password = os.environ.get("OMNI_PASSWORD") or os.environ.get("DB_PASSWORD")
        if host and password:
            import sqlalchemy.ext.asyncio

            connstring = f"postgresql+asyncpg://{user}:{password}@{host}:5432/{db_name}"
            async_engine = sqlalchemy.ext.asyncio.create_async_engine(connstring)
            engine = AlloyDBEngine.from_engine(async_engine)
        else:
            engine = await AlloyDBEngine.afrom_instance(
                project_id=db_project,
                instance=db_instance,
                cluster=db_cluster,
                region=db_region,
                database=db_name,
            )

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{IMAGE_TABLE}"')
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine):
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def image_vs(self, engine):
        await engine._ainit_vectorstore_table(
            IMAGE_TABLE,
            VECTOR_SIZE,
            metadata_columns=[
                Column("image_id", "TEXT"),
                Column("source", "TEXT"),
            ],
            metadata_json_column="mymeta",
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=image_embedding_service,
            table_name=IMAGE_TABLE,
            metadata_columns=["image_id", "source"],
            metadata_json_column="mymeta",
        )
        yield vs

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

    async def test_init_with_constructor(self, engine):
        with pytest.raises(Exception):
            AsyncAlloyDBVectorStore(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_post_init(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_id_metadata_column(self, engine):
        table_name = "id_metadata" + str(uuid.uuid4())
        await engine._ainit_vectorstore_table(
            table_name,
            VECTOR_SIZE,
            metadata_columns=[Column("id", "TEXT")],
        )
        custom_vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
            metadata_columns=["id"],
        )
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await custom_vs.aadd_texts(texts, id_column_as_metadata, ids)

        results = await afetch(engine, f'SELECT * FROM "{table_name}"')
        assert len(results) == 3
        assert results[0]["id"] == "0"
        assert results[1]["id"] == "1"
        assert results[2]["id"] == "2"
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_aadd_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 6
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_texts_edge_cases(self, engine, vs):
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs_no_ids(self, engine, vs):
        await vs.aadd_documents(docs)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_images(self, engine, image_vs, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        metadatas = [
            {"image_id": str(i), "source": "google.com"} for i in range(len(image_uris))
        ]
        await image_vs.aadd_images(image_uris, metadatas, ids)
        results = await afetch(engine, (f'SELECT * FROM "{IMAGE_TABLE}"'))
        assert len(results) == len(image_uris)
        assert results[0]["image_id"] == "0"
        assert results[0]["source"] == "google.com"
        await aexecute(engine, (f'TRUNCATE TABLE "{IMAGE_TABLE}"'))

    async def test_aadd_images_store_uri_only(self, engine, image_vs, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        metadatas = [
            {"image_id": str(i), "source": "google.com"} for i in range(len(image_uris))
        ]
        await image_vs.aadd_images(image_uris, metadatas, ids, store_uri_only=True)
        results = await afetch(engine, (f'SELECT * FROM "{IMAGE_TABLE}"'))
        assert len(results) == len(image_uris)
        # Check that content column stores the URI
        for i, result_row in enumerate(results):
            assert result_row[image_vs.content_column] == image_uris[i]
            # Check that embedding is not an embedding of the URI string itself (basic check)
            uri_embedding = embeddings_service.embed_query(image_uris[i])
            image_embedding = image_embedding_service.embed_image([image_uris[i]])[0]
            actual_embedding = json.loads(result_row[image_vs.embedding_column])
            assert actual_embedding != pytest.approx(uri_embedding)
            assert actual_embedding == pytest.approx(image_embedding)
            assert result_row["image_id"] == str(i)
            assert result_row["source"] == "google.com"
            # Check that the original URI is also in the metadata (json column)
            assert (
                result_row[image_vs.metadata_json_column]["image_uri"] == image_uris[i]
            )

        await aexecute(engine, (f'TRUNCATE TABLE "{IMAGE_TABLE}"'))

    async def test_adelete(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        # delete an ID
        await vs.adelete([ids[0]])
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2
        # delete with no ids
        result = await vs.adelete()
        assert result == False
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    ##### Custom Vector Store  #####
    async def test_aadd_embeddings(self, engine, vs_custom):
        await vs_custom.aadd_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas
        )
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, metadatas, ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 6
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await vs_custom.aadd_documents(docs, ids=ids)

        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_adelete_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        # delete an ID
        await vs_custom.adelete([ids[0]])
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 2
        assert "foo" not in content
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_ignore_metadata_columns(self, engine):
        column_to_ignore = "source"
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            ignore_metadata_columns=[column_to_ignore],
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_json_column="mymeta",
        )
        assert column_to_ignore not in vs.metadata_columns

    async def test_create_vectorstore_with_invalid_parameters_1(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )

    async def test_create_vectorstore_with_invalid_parameters_2(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="langchain_id",  # invalid content column type
                embedding_column="myembedding",
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_3(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="random_column",  # invalid embedding column
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_4(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",  # invalid embedding column data type
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_5(self, engine):
        with pytest.raises(ValueError):
            await AsyncAlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",
                metadata_columns=["random_column"],
                ignore_metadata_columns=[
                    "one",
                    "two",
                ],  # invalid use of metadata_columns and ignore columns
            )

    async def test_create_vectorstore_with_init(self, engine):
        with pytest.raises(Exception):
            await AsyncAlloyDBVectorStore(
                engine._pool,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )

    async def test_live_columnar_engine(self, vs):
        """Test enabling columnar engine against live AlloyDB instance."""
        await vs.aenable_columnar_engine(["content"])
        await vs.aenable_columnar_engine()

        # Assert functional similarity search still works on columnarized table
        await vs.aadd_texts(["Columnar engine test document"])
        results = await vs.asimilarity_search("Columnar test", k=1)
        assert len(results) > 0
        assert "Columnar" in results[0].page_content

    async def test_live_auto_columnarization(self, vs):
        """Test triggering auto columnarization recommendations against live AlloyDB instance."""
        try:
            await vs.aenable_auto_columnarization()
        except Exception as e:
            if "google_columnar_engine.enabled" in str(
                e
            ) or "shared_preload_libraries" in str(e):
                pytest.skip(f"Columnar engine flag not enabled on instance: {e}")
            raise

        # Assert functional similarity search still works after auto columnarization
        await vs.aadd_texts(["Auto columnarization test document"])
        results = await vs.asimilarity_search("Auto columnarization", k=1)
        assert len(results) > 0
        assert "Auto columnarization" in results[0].page_content

    async def test_live_vector_assist(self, engine):
        """Test vector assist spec definition, application, and recommendations against live AlloyDB instance."""
        table_name = "va_live_table_" + str(uuid.uuid4()).replace("-", "_")
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
        await aexecute(
            engine,
            f"""
            CREATE TABLE "{table_name}" (
                langchain_id uuid PRIMARY KEY,
                content text,
                embedding vector({VECTOR_SIZE}),
                meta jsonb
            );
            """,
        )
        await aexecute(
            engine,
            f"""
            INSERT INTO "{table_name}" (langchain_id, content, embedding, meta)
            SELECT 
                gen_random_uuid(),
                'Content ' || i,
                (SELECT array_agg((random() * 2 - 1)::float4)::vector({VECTOR_SIZE}) FROM generate_series(1, {VECTOR_SIZE})),
                '{{"page": 1}}'::jsonb
            FROM generate_series(1, 100) AS i;
            """,
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
            metadata_json_column="meta",
        )
        specs = await vs.adefine_vector_assist_spec()
        assert isinstance(specs, list)
        assert len(specs) > 0
        apply_res = await vs.aapply_vector_assist_spec()
        assert isinstance(apply_res, list)
        recs = await vs.aget_vector_assist_recommendations()
        assert isinstance(recs, list)
        await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')


@pytest.mark.asyncio
class TestAsyncVectorStoreUnit:
    @pytest.fixture
    def vs(self):
        vs = AsyncAlloyDBVectorStore.__new__(AsyncAlloyDBVectorStore)
        vs.engine = MagicMock()
        vs.schema_name = "public"
        vs.table_name = "test_table"
        vs.content_column = "content"
        vs.embedding_column = "embedding"
        return vs

    async def test_aenable_columnar_engine(self, vs):
        """Test enabling columnar engine with a specific column list."""
        # 1. Mock the database connection
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Call aenable_columnar_engine with a specific column list
            await vs.aenable_columnar_engine(["content"])

            # 3. Assert exact SQL signature and parameters sent to the database
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
            )
            assert call_args[0][1] == {
                "table_name": '"public"."test_table"',
                "columns": '"content"',
            }

    async def test_aenable_columnar_engine_without_columns(self, vs):
        """Test enabling columnar engine without specifying columns (entire table)."""
        # 1. Mock the database connection
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [("content",), ("langchain_id",)]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Call aenable_columnar_engine without column arguments
            await vs.aenable_columnar_engine()

            # 3. Assert default single-argument query is executed
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
            )
            assert call_args[0][1] == {
                "table_name": '"public"."test_table"',
                "columns": '"content","langchain_id"',
            }

    async def test_aenable_auto_columnarization(self, vs):
        """Test enabling auto columnarization executes queries on engine."""
        # 1. Mock the database connection
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Trigger auto columnarization recommendations
            await vs.aenable_auto_columnarization()

            # 3. Assert recommendation query executed on engine
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "SELECT google_columnar_engine_recommend('AUTO_COLUMNARIZATION')"
            )

    async def test_adefine_vector_assist_spec(self, vs):
        """Test definition of vector assist specification."""
        # 1. Mock database returning a vector assist spec row
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [{"spec": "ok"}]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Call adefine_vector_assist_spec
            res = await vs.adefine_vector_assist_spec()

            # 3. Assert returned spec list and query parameters
            assert res == [{"spec": "ok"}]
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "SELECT * FROM vector_assist.define_spec(table_name => :table_name, vector_column_name => :embedding_column)"
            )
            assert call_args[0][1] == {
                "table_name": '"public"."test_table"',
                "embedding_column": "embedding",
            }

    async def test_aapply_vector_assist_spec(self, vs):
        """Test applying vector assist specifications."""
        # 1. Mock database applying vector assist spec
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.mappings.return_value = [{"apply": "ok"}]
            mock_conn.execute.return_value = mock_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Apply spec
            res = await vs.aapply_vector_assist_spec()

            # 3. Assert results and query parameters
            assert res == [{"apply": "ok"}]
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "SELECT * FROM vector_assist.apply_spec(table_name => :table_name, vector_column_name => :embedding_column)"
            )
            assert call_args[0][1] == {
                "table_name": '"public"."test_table"',
                "embedding_column": "embedding",
            }

    async def test_aget_vector_assist_recommendations(self, vs):
        """Test retrieving vector assist recommendations with a valid spec ID."""
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            # First query returns spec_id, second query returns recommendations
            mock_spec_result = MagicMock()
            mock_spec_result.mappings.return_value.first.return_value = {
                "spec_id": "spec123"
            }
            mock_rec_result = MagicMock()
            mock_rec_result.mappings.return_value = [{"rec": "ok"}]
            mock_conn.execute.side_effect = [mock_spec_result, mock_rec_result]
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Retrieve recommendations
            res = await vs.aget_vector_assist_recommendations()

            # 3. Assert recommendations and query calls
            assert res == [{"rec": "ok"}]
            assert mock_conn.execute.call_count == 2

    async def test_aget_vector_assist_recommendations_empty_specs(self, vs):
        """Test retrieving vector assist recommendations when no specs exist in vector_assist.specs."""
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_spec_result = MagicMock()
            mock_spec_result.mappings.return_value.first.return_value = None
            mock_conn.execute.return_value = mock_spec_result
            mock_connect.return_value.__aenter__.return_value = mock_conn

            res = await vs.aget_vector_assist_recommendations()
            assert res == []

    async def test_ainitialize_auto_vector_embeddings(self, vs):
        """Test initializing auto vector embeddings asynchronously with default columns."""
        # 1. Mock the database connection
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Call auto vector embedding initialization
            await vs.ainitialize_auto_vector_embeddings(
                model_id="test-model",
            )

            # 3. Assert exact procedure call and parameters
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
            )
            assert call_args[0][1] == {
                "model_id": "test-model",
                "table_name": '"public"."test_table"',
                "content_column": "content",
                "embedding_column": "embedding",
            }

    async def test_ainitialize_auto_vector_embeddings_custom_columns(self, vs):
        """Test initializing auto vector embeddings with custom columns and schema."""
        # 1. Mock the database connection
        with patch.object(vs.engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            # 2. Call with custom content column, embedding column, and schema
            await vs.ainitialize_auto_vector_embeddings(
                model_id="test-model",
                content_column="custom_content",
                embedding_column="custom_embedding",
                schema_name="myschema",
            )

            # 3. Assert custom parameters and quoted schema identifier
            call_args = mock_conn.execute.call_args
            assert (
                str(call_args[0][0])
                == "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
            )
            assert call_args[0][1] == {
                "model_id": "test-model",
                "table_name": '"myschema"."test_table"',
                "content_column": "custom_content",
                "embedding_column": "custom_embedding",
            }

    async def test_ainitialize_auto_vector_embeddings_missing_columns(self, vs):
        """Test error raised when required content column name is missing."""
        # 1. Clear content_column on vector store
        vs.content_column = None

        # 2. Assert ValueError is raised when calling without content_column
        with pytest.raises(
            ValueError, match="content_column must be provided or configured"
        ):
            await vs.ainitialize_auto_vector_embeddings(model_id="test-model")

    async def test_ainitialize_auto_vector_embeddings_missing_embedding_column(
        self, vs
    ):
        """Test error raised when required embedding_column name is missing."""
        # 1. Clear embedding_column on vector store
        vs.embedding_column = None

        # 2. Assert ValueError is raised when calling without embedding_column
        with pytest.raises(
            ValueError, match="embedding_column must be provided or configured"
        ):
            await vs.ainitialize_auto_vector_embeddings(model_id="test-model")
