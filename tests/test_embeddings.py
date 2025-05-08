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
        return "text-embedding-005"

    @pytest_asyncio.fixture
    async def embeddings(self, engine, model_id):
        model_manager = await AlloyDBModelManager.create(engine=engine)
        model = await model_manager.aget_model(model_id=model_id)
        if not model:
            # create model if not exists
            await model_manager.acreate_model(
                model_id=model_id,
                model_provider="google",
                model_qualified_name=model_id,  # assuming model is built-in
                model_type="text_embedding",
            )
        return AlloyDBEmbeddings.create_sync(engine=engine, model_id=model_id)

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

    async def test_embed_query_inline(self, embeddings, model_id):
        embedding_query = embeddings.embed_query_inline("test document")
        assert embedding_query == f"embedding('{model_id}', 'test document')::vector"

    async def test_aembed_query(self, embeddings):
        embedding = await embeddings.aembed_query("test document")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for embedding_field in embedding:
            assert isinstance(embedding_field, float)
            assert -1 <= embedding_field <= 1
