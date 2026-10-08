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
from sqlalchemy.exc import DBAPIError

from langchain_google_alloydb_pg import (
    AlloyDBEmbeddings,
    AlloyDBEngine,
    AlloyDBModelManager,
)

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "test-table" + str(uuid.uuid4())
embedding_model = "text-embedding-005" + str(uuid.uuid4()).replace("-", "_")
multimodal_embedding_model = "multimodalembedding@001"
# 1x1 RGB PNG encoded as base64 for live google_ml.image_embedding tests.
TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB"
    "7CYF4AAAAAElFTkSuQmCC"
)
MULTIMODAL_EMBEDDING_DIM = 1408


@pytest.mark.asyncio
class TestAlloyDBEmbeddings:

    @pytest_asyncio.fixture
    async def engine(self):
        AlloyDBEngine._connector = None
        engine = await AlloyDBEngine.afrom_instance(
            project_id=project_id,
            cluster=cluster_id,
            instance=instance_id,
            region=region,
            database=db_name,
        )
        yield engine

        await engine.close()

    @pytest_asyncio.fixture
    async def sync_engine(self):
        AlloyDBEngine._connector = None
        engine = AlloyDBEngine.from_instance(
            project_id=project_id,
            cluster=cluster_id,
            instance=instance_id,
            region=region,
            database=db_name,
        )
        yield engine

        await engine.close()

    @pytest.fixture(scope="module")
    def model_id(self) -> str:
        return embedding_model

    @pytest_asyncio.fixture
    async def embeddings(self, engine, model_id):
        model_manager = await AlloyDBModelManager.create(engine=engine)
        model = await model_manager.aget_model(model_id=model_id)
        if not model:
            # create model if not exists
            await model_manager.acreate_model(
                model_id=model_id,
                model_provider="google",
                model_qualified_name="text-embedding-005",  # assuming model is built-in
                model_type="text_embedding",
            )
        return AlloyDBEmbeddings.create_sync(engine=engine, model_id=model_id)

    @pytest_asyncio.fixture
    async def multimodal_embeddings(self, engine):
        model_manager = await AlloyDBModelManager.create(engine=engine)
        model = await model_manager.aget_model(model_id=multimodal_embedding_model)
        if not model:
            await model_manager.acreate_model(
                model_id=multimodal_embedding_model,
                model_provider="google",
                model_qualified_name=multimodal_embedding_model,
                model_type="multimodal_embedding",
            )
        return await AlloyDBEmbeddings.create(
            engine=engine, model_id=multimodal_embedding_model
        )

    async def test_model_exists(self, sync_engine):
        test_model_id = "test_sample_text_embedding_model"
        error_message = f"Model {test_model_id} does not exist."
        with pytest.raises(Exception, match=error_message):
            AlloyDBEmbeddings.create_sync(engine=sync_engine, model_id=test_model_id)

    async def test_amodel_exists(self, engine):
        test_model_id = "test_sample_text_embedding_model"
        error_message = f"Model {test_model_id} does not exist."
        with pytest.raises(Exception, match=error_message):
            await AlloyDBEmbeddings.create(engine=engine, model_id=test_model_id)

    async def test_aembed_documents(self, embeddings):
        with pytest.raises(NotImplementedError):
            await embeddings.aembed_documents([Document(page_content="test document")])

    async def test_embed_documents(self, embeddings):
        with pytest.raises(NotImplementedError):
            embeddings.embed_documents([Document(page_content="test document")])

    async def test_embed_query(self, embeddings):
        embedding = embeddings.embed_query("test document")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    async def test_embed_query_sql_injection(self, embeddings):
        malicious_query = "'); DROP TABLE users; --"
        embedding = embeddings.embed_query(malicious_query)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    async def test_embed_query_inline(self, embeddings, model_id):
        embedding_query = embeddings.embed_query_inline("test document")
        assert embedding_query == f"embedding('{model_id}', 'test document')::vector"

    async def test_embed_query_inline_template(self, embeddings, model_id):
        embedding_query = embeddings.embed_query_inline_template(":content")
        assert embedding_query == f"embedding('{model_id}', :content)::vector"
        embedding_query_search = embeddings.embed_query_inline_template(":query_text")
        assert embedding_query_search == f"embedding('{model_id}', :query_text)::vector"
        with pytest.raises(ValueError, match="Invalid parameter name"):
            embeddings.embed_query_inline_template("invalid_no_colon")
        with pytest.raises(ValueError, match="Invalid parameter name"):
            embeddings.embed_query_inline_template(":content); DROP TABLE users; --")

    async def test_embed_query_inline_sql_injection(self, embeddings, model_id):
        malicious_query = "'); DROP TABLE users; --"
        embedding_query = embeddings.embed_query_inline(malicious_query)
        assert (
            embedding_query
            == f"embedding('{model_id}', '''); DROP TABLE users; --')::vector"
        )

    async def test_aembed_query(self, embeddings):
        embedding = await embeddings.aembed_query("test document")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    async def test_aembed_query_sql_injection(self, embeddings):
        malicious_query = "'); DROP TABLE users; --"
        embedding = await embeddings.aembed_query(malicious_query)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    async def test_embed_image(self, multimodal_embeddings):
        embedding = multimodal_embeddings.embed_image(TEST_IMAGE_B64)
        assert isinstance(embedding, list)
        assert len(embedding) == MULTIMODAL_EMBEDDING_DIM
        assert any(v != 0.0 for v in embedding)
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    async def test_aembed_image(self, multimodal_embeddings):
        embedding = await multimodal_embeddings.aembed_image(TEST_IMAGE_B64)
        assert isinstance(embedding, list)
        assert len(embedding) == MULTIMODAL_EMBEDDING_DIM
        assert any(v != 0.0 for v in embedding)
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1

    @pytest.mark.parametrize("bad_uri", ["", "   "])
    async def test_embed_image_validation_error(self, multimodal_embeddings, bad_uri):
        with pytest.raises(ValueError, match="image_uri must be a non-empty string"):
            multimodal_embeddings.embed_image(bad_uri)
        with pytest.raises(ValueError, match="image_uri must be a non-empty string"):
            await multimodal_embeddings.aembed_image(bad_uri)

    async def test_embed_image_wrong_model_type_raises(self, embeddings):
        with pytest.raises(DBAPIError, match="not a multimodal embedding model"):
            embeddings.embed_image(TEST_IMAGE_B64)
        with pytest.raises(DBAPIError, match="not a multimodal embedding model"):
            await embeddings.aembed_image(TEST_IMAGE_B64)

    async def _assert_image_sql_injection_blocked(self, engine, run_call):
        temp_table = "temp_" + str(uuid.uuid4()).replace("-", "_")

        async def _setup():
            async with engine._pool.connect() as conn:
                await conn.execute(
                    text(f"CREATE TABLE {temp_table} (id INT, val TEXT)")
                )
                await conn.execute(
                    text(f"INSERT INTO {temp_table} VALUES (1, 'untouched')")
                )
                await conn.commit()

        async def _check():
            async with engine._pool.connect() as conn:
                res = await conn.execute(
                    text(f"SELECT val FROM {temp_table} WHERE id = 1")
                )
                return [dict(r) for r in res.mappings()]

        async def _cleanup():
            async with engine._pool.connect() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
                await conn.commit()

        await engine._run_as_async(_setup())
        try:
            malicious_uri = f"'); DROP TABLE {temp_table}; --"
            with pytest.raises(DBAPIError) as exc_info:
                await run_call(malicious_uri)
            assert getattr(exc_info.value.orig, "sqlstate", None) == "GAV03"
            rows = await engine._run_as_async(_check())
            assert rows == [{"val": "untouched"}]
        finally:
            await engine._run_as_async(_cleanup())

    async def test_embed_image_sql_injection(self, engine, multimodal_embeddings):
        async def _call(uri: str) -> None:
            multimodal_embeddings.embed_image(uri)

        await self._assert_image_sql_injection_blocked(engine, _call)

    async def test_aembed_image_sql_injection(self, engine, multimodal_embeddings):
        await self._assert_image_sql_injection_blocked(
            engine, multimodal_embeddings.aembed_image
        )
