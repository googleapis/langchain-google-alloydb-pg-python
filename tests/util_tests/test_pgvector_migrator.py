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
from typing import List, Optional, Sequence

import pytest
import pytest_asyncio
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from sqlalchemy import RowMapping, text

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column
from langchain_google_alloydb_pg.utils.pgvector_migrator import (
    aextract_pgvector_collection,
    alist_pgvector_collection_names,
    amigrate_pgvector_collection,
    extract_pgvector_collection,
    list_pgvector_collection_names,
    migrate_pgvector_collection,
)

COLLECTIONS_TABLE = "langchain_pg_collection"
EMBEDDINGS_TABLE = "langchain_pg_embedding"
VECTOR_SIZE = 768
COLLECTION_NAME_SUFFIX = str(uuid.uuid4()).replace("-", "_")


async def aexecute(
    engine: AlloyDBEngine, query: str, params: Optional[dict] = None
) -> None:
    async def run(engine, query, params):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    await engine._run_as_async(run(engine, query, params))


async def afetch(engine: AlloyDBEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestPgvectorengine:
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

    @pytest_asyncio.fixture(scope="module", params=["PUBLIC"])
    async def engine(
        self,
        request,
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
            database=db_name,
            cluster=db_cluster,
            ip_type=request.param,
            user=user,
            password=password,
        )
        await aexecute(
            engine,
            query=f"CREATE table {COLLECTIONS_TABLE} (uuid VARCHAR, name VARCHAR, cmetadata JSONB)",
        )
        await aexecute(
            engine,
            query=f"CREATE table {EMBEDDINGS_TABLE} (id VARCHAR, collection_id VARCHAR, embedding vector(768), document TEXT, cmetadata JSONB)",
        )
        yield engine
        await aexecute(engine, f"DROP TABLE {COLLECTIONS_TABLE}")
        await aexecute(engine, f"DROP TABLE {EMBEDDINGS_TABLE}")

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

    async def _create_collection(
        self,
        engine: AlloyDBEngine,
        collection_name: str,
        sample_embeddings: List[float],
        num_rows: int = 2,
        num_cols: int = 3,
    ) -> None:
        collection_id = f"collection_id_{collection_name}"
        await aexecute(
            engine,
            query=f"INSERT INTO {COLLECTIONS_TABLE} (uuid, name) VALUES (:uuid, :collection_name)",
            params={"uuid": collection_id, "collection_name": collection_name},
        )
        for row_num in range(num_rows):
            await aexecute(
                engine,
                query=f"""INSERT INTO {EMBEDDINGS_TABLE} (id, collection_id, embedding, document, cmetadata) VALUES (:id, :collection_id, :embedding, :document, :cmetadata)""",
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

    async def _create_pgvector_tables(
        self,
        engine: AlloyDBEngine,
        sample_embeddings: List[float],
        num_rows: int = 2,
        num_collections: int = 1,
        num_cols: int = 3,
    ) -> None:
        """Create embeddings as well as collections table."""
        for collection_num in range(num_collections):
            collection_name = f"collection_{collection_num}_{COLLECTION_NAME_SUFFIX}"
            await self._create_collection(
                engine,
                collection_name,
                sample_embeddings,
                num_rows=num_rows,
                num_cols=num_cols,
            )

    async def _collect_async_items(self, batch_docs_generator):
        """Collects items from an async generator."""
        docs = []
        async for doc in batch_docs_generator:
            docs.extend(doc)
        return docs

    def _collect_sync_items(self, batch_docs_generator):
        """Collects items from an async generator."""
        docs = []
        for doc in batch_docs_generator:
            docs.extend(doc)
        return docs

    async def _clean_tables(self, engine):
        await aexecute(engine, f"TRUNCATE TABLE {EMBEDDINGS_TABLE}")
        await aexecute(engine, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    #### Async tests
    async def test_aextract_pgvector_collection_exists(self, engine, sample_embeddings):
        await self._create_pgvector_tables(engine, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        results = await self._collect_async_items(
            aextract_pgvector_collection(engine, collection_name)
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

        await self._clean_tables(engine)

    async def test_aextract_pgvector_collection_non_existant(self, engine):
        collection_name = "random_collection"
        with pytest.raises(ValueError):
            await self._collect_async_items(
                aextract_pgvector_collection(engine, collection_name)
            )
        await self._clean_tables(engine)

    async def test_amigrate_pgvector_collection_json_metadata(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = await AlloyDBVectorStore.create(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
        )
        await amigrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_col_metadata(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        metadata_columns = [
            Column(f"col_0_{collection_name}", "VARCHAR"),
            Column(f"col_1_{collection_name}", "VARCHAR"),
        ]
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            metadata_columns=metadata_columns,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = await AlloyDBVectorStore.create(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
            metadata_columns=[col.name for col in metadata_columns],
        )
        await amigrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, col_0_{collection_name}, col_1_{collection_name}, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            f"col_0_{collection_name}": f"val_0_{collection_name}",
            f"col_1_{collection_name}": f"val_0_{collection_name}",
            "langchain_metadata": {
                f"col_2_{collection_name}": f"val_0_{collection_name}"
            },
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_delete_original(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = await AlloyDBVectorStore.create(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
        )
        await amigrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
            delete_pg_collection=True,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 0}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 0}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_batch(self, engine, sample_embeddings):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=7)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = await AlloyDBVectorStore.create(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
        )
        await amigrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
            insert_batch_size=5,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 7}]

        # Check last row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name}",
        )
        expected_row = {
            "langchain_id": f"uuid_6_{collection_name}",
            "content": "content_6",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 6, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_alist_pgvector_collection_names(self, engine, sample_embeddings):
        num_collections = 3
        await self._create_pgvector_tables(
            engine, sample_embeddings, num_collections=num_collections
        )
        all_collections = await alist_pgvector_collection_names(engine)
        assert len(all_collections) == num_collections

        expected = [
            f"collection_{i}_{COLLECTION_NAME_SUFFIX}" for i in range(num_collections)
        ]
        for collection in all_collections:
            assert collection in expected

        await self._clean_tables(engine)

    async def test_alist_pgvector_collection_names_error(self, engine):
        await aexecute(engine, f"DROP TABLE IF EXISTS {COLLECTIONS_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {EMBEDDINGS_TABLE}")
        with pytest.raises(ValueError):
            await alist_pgvector_collection_names(engine)
        await aexecute(
            engine,
            query=f"CREATE table {COLLECTIONS_TABLE} (uuid VARCHAR, name VARCHAR, cmetadata JSONB)",
        )
        await aexecute(
            engine,
            query=f"CREATE table {EMBEDDINGS_TABLE} (id VARCHAR, collection_id VARCHAR, embedding vector(768), document TEXT, cmetadata JSONB)",
        )

    #### Sync tests
    async def test_extract_pgvector_collection_exists(self, engine, sample_embeddings):
        await self._create_pgvector_tables(engine, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        results = self._collect_sync_items(
            extract_pgvector_collection(engine, collection_name)
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

        await self._clean_tables(engine)

    async def test_extract_pgvector_collection_non_existant(self, engine):
        collection_name = "random_collection"
        with pytest.raises(ValueError):
            self._collect_sync_items(
                extract_pgvector_collection(engine, collection_name)
            )
        await self._clean_tables(engine)

    async def test_migrate_pgvector_collection_json_metadata(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"

        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = AlloyDBVectorStore.create_sync(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
        )
        migrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_col_metadata(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        metadata_columns = [
            Column(f"col_0_{collection_name}", "VARCHAR"),
            Column(f"col_1_{collection_name}", "VARCHAR"),
        ]
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            metadata_columns=metadata_columns,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = AlloyDBVectorStore.create_sync(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
            metadata_columns=[col.name for col in metadata_columns],
        )
        migrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, col_0_{collection_name}, col_1_{collection_name}, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            f"col_0_{collection_name}": f"val_0_{collection_name}",
            f"col_1_{collection_name}": f"val_0_{collection_name}",
            "langchain_metadata": {
                f"col_2_{collection_name}": f"val_0_{collection_name}"
            },
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_delete_original(
        self, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = AlloyDBVectorStore.create_sync(
            engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
            table_name=collection_name,
        )
        migrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
            delete_pg_collection=True,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 0, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # The collection data should be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embeddings_table_count = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embeddings_table_count == [{"count": 0}]
        collection_table_entry = await afetch(
            engine,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 0}]

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_batch(self, engine, sample_embeddings):
        # Set up tables
        await self._create_pgvector_tables(engine, sample_embeddings, num_rows=7)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        vector_store = AlloyDBVectorStore.create_sync(
            table_name=collection_name,
            engine=engine,
            embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        )
        migrate_pgvector_collection(
            engine,
            collection_name=collection_name,
            vector_store=vector_store,
            insert_batch_size=5,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            engine, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 7}]

        # Check last row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            engine,
            f"SELECT langchain_id, content, embedding, langchain_metadata FROM {collection_name}",
        )
        expected_row = {
            "langchain_id": f"uuid_6_{collection_name}",
            "content": "content_6",
            "embedding": str(sample_embeddings).replace(" ", ""),
            "langchain_metadata": self._create_metadata_for_collection(
                collection_name, 6, num_cols=3
            ),
        }
        assert expected_row in migrated_data

        # Delete set up tables
        await self._clean_tables(engine)
        await aexecute(engine, f"DROP TABLE {collection_name}")

    async def test_list_pgvector_collection_names(self, engine, sample_embeddings):
        num_collections = 3
        await self._create_pgvector_tables(
            engine, sample_embeddings, num_collections=num_collections
        )
        all_collections = list_pgvector_collection_names(engine)
        assert len(all_collections) == num_collections

        expected = [
            f"collection_{i}_{COLLECTION_NAME_SUFFIX}" for i in range(num_collections)
        ]
        for collection in all_collections:
            assert collection in expected

        await self._clean_tables(engine)

    async def test_list_pgvector_collection_names_error(self, engine):
        await aexecute(engine, f"DROP TABLE IF EXISTS {EMBEDDINGS_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {COLLECTIONS_TABLE}")
        with pytest.raises(ValueError):
            list_pgvector_collection_names(engine)
        await aexecute(
            engine,
            query=f"CREATE table {COLLECTIONS_TABLE} (uuid VARCHAR, name VARCHAR, cmetadata JSONB)",
        )
        await aexecute(
            engine,
            query=f"CREATE table {EMBEDDINGS_TABLE} (id VARCHAR, collection_id VARCHAR, embedding vector(768), document TEXT, cmetadata JSONB)",
        )
