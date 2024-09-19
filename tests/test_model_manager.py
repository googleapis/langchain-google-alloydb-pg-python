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

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBModelManager


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


EMBEDDING_MODEL_NAME = "textembedding-gecko@003" + str(uuid.uuid4()).replace("-", "_")


@pytest.mark.asyncio
class TestAlloyDBModelManager:
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

    @pytest_asyncio.fixture(scope="module")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="module")
    async def model(self, engine):
        model = AlloyDBModelManager(engine)
        return model

    async def test_acreate_model(self, model):
        await model.acreate_model(
            model_id=EMBEDDING_MODEL_NAME,  # consider a uuid
            model_provider="google",
            model_qualified_name="textembedding-gecko@003",
            model_type="text_embedding",
        )

    @pytest.mark.depends(on=["test_acreate_model"])
    async def test_alist_model(self, model):
        model_info = await model.alist_model(model_id=EMBEDDING_MODEL_NAME)
        assert len(model_info) == 1
        assert model_info[0]["model_id"] == EMBEDDING_MODEL_NAME

    @pytest.mark.depends(on=["test_acreate_model"])
    async def test_amodel_info_view(self, model):
        models_list = await model.amodel_info_view()
        assert len(models_list) >= 3
        model_ids = [model_info["model_id"] for model_info in models_list]
        assert EMBEDDING_MODEL_NAME in model_ids

    @pytest.mark.depends(on=["test_acreate_model"])
    async def test_adrop_model(self, model):
        await model.adrop_model(model_id=EMBEDDING_MODEL_NAME)
