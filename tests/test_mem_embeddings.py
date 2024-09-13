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
from typing import Sequence

import pytest
import pytest_asyncio

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBMemEmbeddings

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "test-table" + str(uuid.uuid4())


@pytest.mark.asyncio
class TestAlloyDBMemEmbeddings:

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
        return "textembedding-gecko@003"

    @pytest_asyncio.fixture
    def mem_embeddings(self, engine, model_id):
        return AlloyDBMemEmbeddings(engine=engine, model_id=model_id)

    async def test_aembed_documents(self, mem_embeddings):
        with pytest.raises(NotImplementedError):
            await mem_embeddings.aembed_documents(["test document"])

    async def test_embed_documents(self, mem_embeddings):
        with pytest.raises(NotImplementedError):
            mem_embeddings.embed_documents(["test document"])

    async def test_embed_query(self, mem_embeddings):
        with pytest.raises(NotImplementedError):
            mem_embeddings.embed_query("test document")

    async def test_embed_query_inline(self, mem_embeddings):
        embedding_query = mem_embeddings.embed_query_inline("test document")
        assert (
            embedding_query
            == f"embedding('{mem_embeddings.model_id}', 'test document')::vector"
        )

    async def test_aembed_query(self, mem_embeddings):
        embedding = await mem_embeddings.aembed_query("test document")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
