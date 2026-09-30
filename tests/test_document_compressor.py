# Copyright 2026 Google LLC
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
from typing import Optional, Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from langchain_google_alloydb_pg import AlloyDBDocumentCompressor, AlloyDBEngine
from langchain_google_alloydb_pg.document_compressor import _to_ranked_documents

RANKING_MODEL_ID = "semantic-ranker-default-003"
# SQLSTATE AlloyDB AI returns when the instance's service agent may not call
# the Vertex AI ranking API (it needs roles/discoveryengine.viewer).
RANKING_PERMISSION_DENIED = "GAV07"
# google_ml.rank raises (SQLSTATE P0001) unless this database flag is on.
AI_QUERY_ENGINE_FLAG = "google_ml_integration.enable_ai_query_engine"
QUERY = "neural networks"
DOCUMENTS = [
    Document(
        page_content="Alpha: Introduction to Machine Learning.",
        metadata={"source": "book_a", "page": 1},
    ),
    Document(
        page_content="Beta: Deep Learning and Neural Networks.",
        metadata={"source": "book_b", "page": 42},
    ),
    Document(
        page_content="Gamma: Advanced Database Systems in Cloud.",
        metadata={"source": "book_c", "page": 100},
    ),
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestAlloyDBDocumentCompressor:
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
    async def ranking(self, engine):
        """Skips the test if google_ml.rank can't be used on this instance."""
        compressor = AlloyDBDocumentCompressor(engine=engine, model_id=RANKING_MODEL_ID)
        try:
            await compressor.acompress_documents(DOCUMENTS[:1], QUERY)
        except DBAPIError as e:
            sqlstate = getattr(e.orig, "sqlstate", None)
            if sqlstate == RANKING_PERMISSION_DENIED:
                pytest.skip(
                    "google_ml.rank is not permitted on this instance "
                    f"(SQLSTATE {RANKING_PERMISSION_DENIED}): the AlloyDB service "
                    "agent needs roles/discoveryengine.viewer."
                )
            if sqlstate == "P0001" and await self.ai_query_engine_flag(engine) != "on":
                pytest.skip(
                    "google_ml.rank is disabled on this instance (SQLSTATE P0001): "
                    f"the {AI_QUERY_ENGINE_FLAG} flag is off."
                )
            raise

    async def ai_query_engine_flag(self, engine: AlloyDBEngine) -> Optional[str]:
        async def fetch() -> Optional[str]:
            async with engine._pool.connect() as conn:
                result = await conn.execute(
                    text("SELECT current_setting(:flag, true)"),
                    {"flag": AI_QUERY_ENGINE_FLAG},
                )
                return result.scalar()

        return await engine._run_as_async(fetch())

    async def test_model_id_is_required(self, engine):
        with pytest.raises(ValidationError, match="model_id"):
            AlloyDBDocumentCompressor(engine=engine)  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="model_id must be a non-empty"):
            AlloyDBDocumentCompressor(engine=engine, model_id="  ")

    def check_ranked(self, result: Sequence[Document], expected_len: int) -> None:
        assert len(result) == expected_len
        assert result[0].page_content == DOCUMENTS[1].page_content
        assert result[0].metadata["source"] == "book_b"
        scores = [doc.metadata["relevance_score"] for doc in result]
        assert all(isinstance(score, float) for score in scores)
        assert scores == sorted(scores, reverse=True)

    async def test_acompress_documents(self, engine, ranking):
        compressor = AlloyDBDocumentCompressor(engine=engine, model_id=RANKING_MODEL_ID)
        result = await compressor.acompress_documents(DOCUMENTS, QUERY)
        self.check_ranked(result, expected_len=3)

    async def test_compress_documents(self, engine, ranking):
        compressor = AlloyDBDocumentCompressor(engine=engine, model_id=RANKING_MODEL_ID)
        result = compressor.compress_documents(DOCUMENTS, QUERY)
        self.check_ranked(result, expected_len=3)

    async def test_top_n(self, engine, ranking):
        compressor = AlloyDBDocumentCompressor(
            engine=engine, model_id=RANKING_MODEL_ID, top_n=1
        )
        result = await compressor.acompress_documents(DOCUMENTS, QUERY)
        self.check_ranked(result, expected_len=1)

    async def test_empty_documents(self, engine):
        compressor = AlloyDBDocumentCompressor(engine=engine, model_id=RANKING_MODEL_ID)
        assert await compressor.acompress_documents([], QUERY) == []
        assert compressor.compress_documents([], QUERY) == []
        # From a plain thread, with no running event loop.
        assert await asyncio.to_thread(compressor.compress_documents, [], QUERY) == []

    async def test_to_ranked_documents(self):
        rows = [(3, 0.4), (1, 0.8), (2, 0.95)]
        result = _to_ranked_documents(DOCUMENTS, rows)
        assert [doc.page_content for doc in result] == [
            DOCUMENTS[1].page_content,
            DOCUMENTS[0].page_content,
            DOCUMENTS[2].page_content,
        ]
        assert result[0].metadata == {
            "source": "book_b",
            "page": 42,
            "relevance_score": 0.95,
        }
