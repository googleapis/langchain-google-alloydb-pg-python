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

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from sqlalchemy import RowMapping, text

from langchain_google_alloydb_pg import AlloyDBEngine, Column, PgvectorMigrator

COLLECTIONS_TABLE = "collections_" + str(uuid.uuid4()).replace("-", "_")
EMBEDDING_TABLE = "embeddings" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768
COLLECTION_NAME_SUFFIX = str(uuid.uuid4()).replace("-", "_")


async def aexecute(
    migrator: PgvectorMigrator, query: str, params: Optional[dict] = None
) -> None:
    async def run(engine, query, params):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    await migrator._run_as_async(run(migrator.engine, query, params))


async def afetch(migrator: PgvectorMigrator, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await migrator._run_as_async(run(migrator.engine, query))


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestPgvectorMigrator:
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
        return engine

    @pytest_asyncio.fixture(scope="module")
    async def migrator(self, engine):
        migrator = PgvectorMigrator(engine)
        await aexecute(
            migrator,
            query=f"CREATE table {COLLECTIONS_TABLE} (uuid VARCHAR, name VARCHAR, cmetadata JSONB)",
        )
        await aexecute(
            migrator,
            query=f"CREATE table {EMBEDDING_TABLE} (id VARCHAR, collection_id VARCHAR, embedding vector(768), document TEXT, cmetadata JSONB)",
        )
        yield migrator
        await aexecute(migrator, f"DROP TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {EMBEDDING_TABLE}")

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
        migrator: PgvectorMigrator,
        collection_name: str,
        sample_embeddings: List[float],
        num_rows: int = 2,
        num_cols: int = 3,
    ) -> None:
        collection_id = f"collection_id_{collection_name}"
        await aexecute(
            migrator,
            query=f"INSERT INTO {COLLECTIONS_TABLE} (uuid, name) VALUES (:uuid, :collection_name)",
            params={"uuid": collection_id, "collection_name": collection_name},
        )
        for row_num in range(num_rows):
            await aexecute(
                migrator,
                query=f"""INSERT INTO {EMBEDDING_TABLE} (id, collection_id, embedding, document, cmetadata) VALUES (:id, :collection_id, :embedding, :document, :cmetadata)""",
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
        migrator: PgvectorMigrator,
        sample_embeddings: List[float],
        num_rows: int = 2,
        num_collections: int = 1,
        num_cols: int = 3,
    ) -> None:
        """Create embeddings as well as collections table."""
        for collection_num in range(num_collections):
            collection_name = f"collection_{collection_num}_{COLLECTION_NAME_SUFFIX}"
            await self._create_collection(
                migrator,
                collection_name,
                sample_embeddings,
                num_rows=num_rows,
                num_cols=num_cols,
            )

    async def test_execute(self, migrator):
        await aexecute(migrator, query="SELECT 1")

    #### Async tests
    async def test_aextract_pgvector_collection_exists(
        self, migrator, sample_embeddings
    ):
        await self._create_pgvector_tables(migrator, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        results = await migrator.aextract_pgvector_collection(
            collection_name, EMBEDDING_TABLE, COLLECTIONS_TABLE
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

        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_aextract_pgvector_collection_non_existant(self, migrator):
        collection_name = "random_collection"
        with pytest.raises(ValueError):
            await migrator.aextract_pgvector_collection(
                collection_name, EMBEDDING_TABLE, COLLECTIONS_TABLE
            )
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_amigrate_pgvector_collection_error(
        self, migrator, sample_embeddings
    ):
        await self._create_pgvector_tables(migrator, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        with pytest.raises(ValueError):
            await migrator.amigrate_pgvector_collection(
                collection_name=collection_name,
                delete_pg_collection=True,
                pg_embedding_table_name=EMBEDDING_TABLE,
                pg_collection_table_name=COLLECTIONS_TABLE,
            )
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE IF EXISTS {collection_name}")

    async def test_amigrate_pgvector_collection_json_metadata(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        await migrator.amigrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_col_metadata(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        metadata_columns = [
            Column(f"col_0_{collection_name}", "VARCHAR"),
            Column(f"col_1_{collection_name}", "VARCHAR"),
            Column(f"col_2_{collection_name}", "VARCHAR"),
        ]
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            metadata_columns=metadata_columns,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        await migrator.amigrate_pgvector_collection(
            collection_name=collection_name,
            metadata_columns=metadata_columns,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
            f"SELECT langchain_id, content, embedding, col_0_{collection_name}, col_1_{collection_name}, col_2_{collection_name} FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            f"col_0_{collection_name}": f"val_0_{collection_name}",
            f"col_1_{collection_name}": f"val_0_{collection_name}",
            f"col_2_{collection_name}": f"val_0_{collection_name}",
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_delete_original(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        await migrator.amigrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            delete_pg_collection=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 0}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 0}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_amigrate_pgvector_collection_batch(
        self, migrator, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=7)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        await engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        await migrator.amigrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            delete_pg_collection=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
            insert_batch_size=5,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 7}]

        # Check last row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_alist_pgvector_collection_names(self, migrator, sample_embeddings):
        num_collections = 3
        await self._create_pgvector_tables(
            migrator, sample_embeddings, num_collections=num_collections
        )
        all_collections = await migrator.alist_pgvector_collection_names(
            pg_collection_table_name=COLLECTIONS_TABLE
        )
        assert len(all_collections) == num_collections

        expected = [
            f"collection_{i}_{COLLECTION_NAME_SUFFIX}" for i in range(num_collections)
        ]
        for collection in all_collections:
            assert collection in expected

        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_alist_pgvector_collection_names_error(self, migrator):
        with pytest.raises(ValueError):
            await migrator.alist_pgvector_collection_names(
                f"langchain_pg_collection12c92u2973923729_{str(uuid.uuid4()).replace('-', '_')}"
            )

    ### Async tests
    async def test_extract_pgvector_collection_exists(
        self, migrator, sample_embeddings
    ):
        await self._create_pgvector_tables(migrator, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        results = migrator.extract_pgvector_collection(
            collection_name, EMBEDDING_TABLE, COLLECTIONS_TABLE
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

        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_extract_pgvector_collection_non_existant(self, migrator):
        collection_name = "random_collection"
        with pytest.raises(ValueError):
            await migrator.extract_pgvector_collection(
                collection_name, EMBEDDING_TABLE, COLLECTIONS_TABLE
            )
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_migrate_pgvector_collection_error(self, migrator, sample_embeddings):
        await self._create_pgvector_tables(migrator, sample_embeddings)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"

        with pytest.raises(ValueError):
            await migrator.migrate_pgvector_collection(
                collection_name=collection_name,
                delete_pg_collection=True,
                pg_embedding_table_name=EMBEDDING_TABLE,
                pg_collection_table_name=COLLECTIONS_TABLE,
            )
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE IF EXISTS {collection_name}")

    async def test_migrate_pgvector_collection_json_metadata(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"

        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        migrator.migrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_col_metadata(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        metadata_columns = [
            Column(f"col_0_{collection_name}", "VARCHAR"),
            Column(f"col_1_{collection_name}", "VARCHAR"),
            Column(f"col_2_{collection_name}", "VARCHAR"),
        ]
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            metadata_columns=metadata_columns,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        migrator.migrate_pgvector_collection(
            collection_name=collection_name,
            metadata_columns=metadata_columns,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
            f"SELECT langchain_id, content, embedding, col_0_{collection_name}, col_1_{collection_name}, col_2_{collection_name} FROM {collection_name} LIMIT 1",
        )
        expected_row = {
            "langchain_id": f"uuid_0_{collection_name}",
            "content": "content_0",
            "embedding": str(sample_embeddings).replace(" ", ""),
            f"col_0_{collection_name}": f"val_0_{collection_name}",
            f"col_1_{collection_name}": f"val_0_{collection_name}",
            f"col_2_{collection_name}": f"val_0_{collection_name}",
        }
        assert expected_row in migrated_data

        # The collection data should not be deleted from both PGVector tables
        collection_id = f"collection_id_{collection_name}"
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 5}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 1}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_delete_original(
        self, engine, migrator, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=5)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        migrator.migrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            delete_pg_collection=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 5}]

        # Check one row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        embedding_table_count = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {EMBEDDING_TABLE} WHERE collection_id = '{collection_id}'",
        )
        assert embedding_table_count == [{"count": 0}]
        collection_table_entry = await afetch(
            migrator,
            f"SELECT COUNT(*) FROM {COLLECTIONS_TABLE} WHERE uuid = '{collection_id}'",
        )
        assert collection_table_entry == [{"count": 0}]

        # Delete set up tables
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_migrate_pgvector_collection_batch(
        self, migrator, engine, sample_embeddings
    ):
        # Set up tables
        await self._create_pgvector_tables(migrator, sample_embeddings, num_rows=7)
        collection_name = f"collection_0_{COLLECTION_NAME_SUFFIX}"
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=VECTOR_SIZE,
            id_column=Column("langchain_id", "VARCHAR"),
        )
        migrator.migrate_pgvector_collection(
            collection_name=collection_name,
            use_json_metadata=True,
            delete_pg_collection=True,
            pg_embedding_table_name=EMBEDDING_TABLE,
            pg_collection_table_name=COLLECTIONS_TABLE,
            insert_batch_size=5,
        )

        # Check that all data has been migrated
        migrated_table_count = await afetch(
            migrator, f"SELECT COUNT(*) FROM {collection_name}"
        )
        assert migrated_table_count == [{"count": 7}]

        # Check last row to ensure that the data is inserted correctly
        migrated_data = await afetch(
            migrator,
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
        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")
        await aexecute(migrator, f"DROP TABLE {collection_name}")

    async def test_list_pgvector_collection_names(self, migrator, sample_embeddings):
        num_collections = 3
        await self._create_pgvector_tables(
            migrator, sample_embeddings, num_collections=num_collections
        )
        all_collections = migrator.list_pgvector_collection_names(
            pg_collection_table_name=COLLECTIONS_TABLE
        )
        assert len(all_collections) == num_collections

        expected = [
            f"collection_{i}_{COLLECTION_NAME_SUFFIX}" for i in range(num_collections)
        ]
        for collection in all_collections:
            assert collection in expected

        await aexecute(migrator, f"TRUNCATE TABLE {EMBEDDING_TABLE}")
        await aexecute(migrator, f"TRUNCATE TABLE {COLLECTIONS_TABLE}")

    async def test_list_pgvector_collection_names_error(self, migrator):
        with pytest.raises(ValueError):
            await migrator.list_pgvector_collection_names(
                f"langchain_pg_collection923729_{str(uuid.uuid4()).replace('-', '_')}"
            )
