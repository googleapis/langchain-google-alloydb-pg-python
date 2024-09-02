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
from typing import List

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from google.cloud.alloydb.connector import AsyncConnector, IPTypes
from langchain_core.embeddings import FakeEmbeddings
from sqlalchemy import VARCHAR
from sqlalchemy.ext.asyncio import create_async_engine

from langchain_google_alloydb_pg import AlloyDBEngine, Column

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = VECTOR_SIZE

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (VECTOR_SIZE - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str = "default") -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (VECTOR_SIZE - 1) + [float(0.0)]


embeddings_service = FakeEmbeddingsWithDimension()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestEngineAsync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "password for AlloyDB")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest_asyncio.fixture(params=["PUBLIC", "PRIVATE"])
    async def engine(
        self, request, db_project, db_region, db_cluster, db_instance, db_name
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            ip_type=request.param,
        )
        yield engine

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await engine._aexecute("SELECT 1")
        await engine._connector.close()
        await engine._engine.dispose()

    async def test_execute(self, engine):
        await engine._aexecute("SELECT 1")

    async def test_init_table(self, engine):
        try:
            await engine._aexecute(f"DROP TABLE {DEFAULT_TABLE}")
        except:
            print("Table already deleted.")

        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        await engine._aexecute(stmt)

        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        await engine._aexecute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init_table_custom(self, engine):
        try:
            await engine._aexecute(f"DROP TABLE {CUSTOM_TABLE}")
        except:
            print("Table already deleted.")

        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[
                Column("page", "TEXT", False),
                Column("source", "TEXT"),
            ],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await engine._afetch(stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

        await engine._aexecute(f"DROP TABLE {CUSTOM_TABLE}")

    async def test_password(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            user=user,
            password=password,
            ip_type="PRIVATE",
        )
        assert engine
        await engine._aexecute("SELECT 1")

    async def test_from_engine(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        async def init_connection_pool(connector):
            async def getconn():
                conn = await connector.connect(  # type: ignore
                    f"projects/{db_project}/locations/{db_region}/clusters/{db_cluster}/instances/{db_instance}",
                    "asyncpg",
                    user=user,
                    password=password,
                    db=db_name,
                    enable_iam_auth=False,
                    ip_type=IPTypes.PRIVATE,
                )
                return conn

            pool = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
            )
            return pool

        async with AsyncConnector() as connector:
            pool = await init_connection_pool(connector)

            engine = AlloyDBEngine.from_engine(pool)
            await engine._aexecute("SELECT 1")

    async def test_column(self):
        with pytest.raises(ValueError):
            Column(32, "VARCHAR")
        with pytest.raises(ValueError):
            Column("test", VARCHAR)


@pytest.mark.asyncio
class TestEngineSync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "password for AlloyDB")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest.fixture(params=["PUBLIC", "PRIVATE"])
    def engine(self, request, db_project, db_region, db_cluster, db_instance, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            cluster=db_cluster,
            ip_type=request.param,
        )
        yield engine

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        engine._execute("SELECT 1")
        await engine._connector.close()
        await engine._engine.dispose()

    def test_execute(self, engine):
        engine._execute("SELECT 1")

    async def test_init_table(self, engine):
        try:
            engine._execute(f"DROP TABLE {DEFAULT_TABLE}")
        except:
            print("Table already deleted.")
        engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        engine._execute(stmt)

        results = engine._fetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        engine._execute(f"DROP TABLE {DEFAULT_TABLE}")

    def test_init_table_custom(self, engine):
        try:
            engine._execute(f"DROP TABLE {CUSTOM_TABLE}")
        except:
            print("Table already deleted.")

        engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = engine._fetch(stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

        engine._execute(f"DROP TABLE {CUSTOM_TABLE}")

    async def test_password(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        user,
        password,
    ):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
            user=user,
            password=password,
        )
        assert engine
        engine._execute("SELECT 1")


class TestEngineMigration:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "user for AlloyDB")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "password for AlloyDB")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest.fixture(params=["PUBLIC", "PRIVATE"])
    def engine(self, request, db_project, db_region, db_cluster, db_instance, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            cluster=db_cluster,
            ip_type=request.param,
        )
        yield engine

    @pytest.fixture(scope="module")
    def sample_embeddings(self) -> List[float]:
        return [0.1] * (VECTOR_SIZE - 1) + [0.2]

    def _create_metadata_for_collection(
        self, collection_name: str, row_num: int, num_cols: int = 3
    ) -> dict:
        metadata = {}
        for col_num in range(num_cols):
            metadata[f"col_{col_num}_{collection_name}"] = (
                f"val_{row_num}_{collection_name}"
            )
        return metadata

    def _create_collection(
        self,
        engine,
        collections_table,
        embedding_table,
        collection_name,
        sample_embeddings,
        num_rows=2,
        num_cols=3,
    ):
        collection_id = f"collection_id_{collection_name}"
        engine._execute(
            query=f"INSERT INTO {collections_table} (uuid, name) VALUES (:uuid, :collection_name)",
            params={"uuid": collection_id, "collection_name": collection_name},
        )
        for row_num in range(num_rows):
            engine._execute(
                query=f"""INSERT INTO {embedding_table} (id, collection_id, embedding, document, cmetadata) VALUES (:id, :collection_id, :embedding, :document, :cmetadata)""",
                params={
                    "collection_id": collection_id,
                    "id": f"uuid_{row_num}_{collection_name}",
                    "embedding": str(sample_embeddings),
                    "document": f"content_{row_num}",
                    "cmetadata": json.dumps(
                        self._create_metadata_for_collection(
                            collection_name, row_num=row_num, num_cols=num_cols
                        )
                    ),
                },
            )

    def _create_pgvector_tables(
        self, engine, sample_embeddings, num_rows=2, num_collections=3, num_cols=3
    ):
        """Create embeddings as well as collections table."""
        embedding_table = f'langchain_pg_embedding{str(uuid.uuid4()).replace("-", "_")}'
        collections_table = (
            f'langchain_pg_collection_{str(uuid.uuid4()).replace("-", "_")}'
        )
        engine._execute(
            query=f"CREATE table {embedding_table} (id VARCHAR, collection_id VARCHAR, embedding vector(768), document TEXT, cmetadata JSONB)"
        )
        engine._execute(
            query=f"CREATE table {collections_table} (uuid VARCHAR, name VARCHAR, cmetadata JSONB)"
        )
        for collection_num in range(num_collections):
            collection_name = f"collection_{collection_num}"
            self._create_collection(
                engine,
                collections_table,
                embedding_table,
                collection_name,
                sample_embeddings,
                num_rows,
                num_cols=num_cols,
            )
        return embedding_table, collections_table

    def test_extract_pgvector_collection_exists(self, engine, sample_embeddings):
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings
        )
        collection_name = "collection_1"
        results = engine.extract_pgvector_collection(
            collection_name, embedding_table, collections_table
        )
        expected = [
            {
                "id": f"uuid_0_{collection_name}",
                "collection_id": f"collection_id_{collection_name}",
                "embedding": str(sample_embeddings).replace(" ", ""),
                "document": "content_0",
                "cmetadata": self._create_metadata_for_collection(
                    collection_name, row_num=0, num_cols=3
                ),
            },
            {
                "id": f"uuid_1_{collection_name}",
                "collection_id": f"collection_id_{collection_name}",
                "embedding": str(sample_embeddings).replace(" ", ""),
                "document": "content_1",
                "cmetadata": self._create_metadata_for_collection(
                    collection_name, row_num=1, num_cols=3
                ),
            },
        ]
        for row in results:
            assert row in expected
        assert len(results) == 2

        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")

    def test_extract_pgvector_collection_non_existant(self, engine, sample_embeddings):
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings
        )
        collection_name = "collection_random"
        with pytest.raises(ValueError):
            engine.extract_pgvector_collection(
                collection_name, embedding_table, collections_table
            )
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")

    def test_get_all_pgvector_collection_names(self, engine, sample_embeddings):
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings
        )
        all_collections = engine.get_all_pgvector_collection_names(
            pg_collection_table_name=collections_table
        )
        assert len(all_collections) == 3
        expected = ["collection_0", "collection_1", "collection_2"]
        for collection in all_collections:
            assert collection in expected
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")

    def test_get_all_pgvector_collection_names_error(self, engine):
        with pytest.raises(ValueError):
            engine.get_all_pgvector_collection_names(
                f"langchain_pg_collection12c_{uuid.uuid4()}"
            )

    def test_migrate_pgvector_collection_error(self, engine, sample_embeddings):
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings
        )
        with pytest.raises(ValueError):
            engine.migrate_pgvector_collection(
                collection_name="collection_1",
                delete_pg_collection=True,
                pg_embedding_table_name=embedding_table,
                pg_collection_table_name=collections_table,
            )
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")

    def test_migrate_pgvector_collection_json_metadata(self, engine, sample_embeddings):
        # Set up tables
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings, num_rows=5, num_collections=3
        )
        collection_name = "collection_1"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=768,
        )
        engine.migrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            pg_embedding_table_name=embedding_table,
            pg_collection_table_name=collections_table,
        )

        # Check that all data has been migrated
        migrated_table_count = engine._fetch(f"SELECT COUNT(*) FROM {collection_name}")
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = engine._fetch(
            f"SELECT content, embedding, langchain_metadata FROM {collection_name} LIMIT 1"
        )
        expected_row = {
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embedding_table_count = engine._fetch(
            f"SELECT COUNT(*) FROM {embedding_table} WHERE collection_id = '{collection_id}'"
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = engine._fetch(
            f"SELECT COUNT(*) FROM {collections_table} WHERE uuid = '{collection_id}'"
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")
        engine._execute(f"DROP TABLE {collection_name}")

    def test_migrate_pgvector_collection_col_metadata(self, engine, sample_embeddings):
        # Set up tables
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings, num_rows=5, num_collections=3, num_cols=3
        )
        collection_name = "collection_1"
        metadata_columns = [
            Column(f"col_0_{collection_name}", "VARCHAR"),
            Column(f"col_1_{collection_name}", "VARCHAR"),
            Column(f"col_2_{collection_name}", "VARCHAR"),
        ]
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=768,
            metadata_columns=metadata_columns,
        )
        engine.migrate_pgvector_collection(
            collection_name=collection_name,
            metadata_columns=metadata_columns,
            pg_embedding_table_name=embedding_table,
            pg_collection_table_name=collections_table,
        )

        # Check that all data has been migrated
        migrated_table_count = engine._fetch(f"SELECT COUNT(*) FROM {collection_name}")
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = engine._fetch(
            f"SELECT content, embedding, col_0_{collection_name}, col_1_{collection_name}, col_2_{collection_name} FROM {collection_name} LIMIT 1"
        )
        expected_row = {
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            f"col_0_{collection_name}": f"val_0_{collection_name}",
            f"col_1_{collection_name}": f"val_0_{collection_name}",
            f"col_2_{collection_name}": f"val_0_{collection_name}",
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embedding_table_count = engine._fetch(
            f"SELECT COUNT(*) FROM {embedding_table} WHERE collection_id = '{collection_id}'"
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = engine._fetch(
            f"SELECT COUNT(*) FROM {collections_table} WHERE uuid = '{collection_id}'"
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")
        engine._execute(f"DROP TABLE {collection_name}")

    def test_migrate_pgvector_collection_delete_original(
        self, engine, sample_embeddings
    ):
        # Set up tables
        embedding_table, collections_table = self._create_pgvector_tables(
            engine, sample_embeddings, num_rows=5, num_collections=3
        )
        collection_name = "collection_1"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=768,
        )
        engine.migrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            delete_pg_collection=True,
            pg_embedding_table_name=embedding_table,
            pg_collection_table_name=collections_table,
        )

        # Check that all data has been migrated
        migrated_table_count = engine._fetch(f"SELECT COUNT(*) FROM {collection_name}")
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = engine._fetch(
            f"SELECT content, embedding, langchain_metadata FROM {collection_name} LIMIT 1"
        )
        expected_row = {
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embedding_table_count = engine._fetch(
            f"SELECT COUNT(*) FROM {embedding_table} WHERE collection_id = '{collection_id}'"
        )
        assert embedding_table_count == [{"count": 0}]
        collection_table_entry = engine._fetch(
            f"SELECT COUNT(*) FROM {collections_table} WHERE uuid = '{collection_id}'"
        )
        assert collection_table_entry == [{"count": 0}]

        # Delete set up tables
        engine._execute(f"DROP TABLE {embedding_table}")
        engine._execute(f"DROP TABLE {collections_table}")
        engine._execute(f"DROP TABLE {collection_name}")
