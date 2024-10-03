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
import asyncio
import warnings
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, TypeVar

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import ProgrammingError

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

COLLECTIONS_TABLE = "langchain_pg_collection"
EMBEDDINGS_TABLE = "langchain_pg_embedding"

T = TypeVar("T")


async def _aget_collection_uuid(
    engine: AlloyDBEngine,
    collection_name: str,
) -> str:
    """
    Get the collection uuid for a collection present in PGVector tables.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The name of the collection to get the uuid for.
    Returns:
        The uuid corresponding to the collection.
    """
    query = (
        f"SELECT name, uuid FROM {COLLECTIONS_TABLE} WHERE name = '{collection_name}'"
    )
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchone()
    if not result_fetch:
        raise ValueError(f"Collection, {collection_name} not found.")
    return result_fetch.uuid


async def _aextract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> AsyncIterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.

    Yields:
        The data present in the collection.
    """
    uuid = await _aget_collection_uuid(engine, collection_name)
    try:
        query = f"SELECT * FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{uuid}'"
        async with engine._pool.connect() as conn:
            result_proxy = await conn.execute(text(query))
            while True:
                rows = result_proxy.fetchmany(size=batch_size)
                if not rows:
                    break
                yield [row._mapping for row in rows]
    except:
        raise ValueError(f"Collection, {collection_name} does not exist.")


async def _amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    destination_table: Optional[str] = None,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (AlloyDBVectorStore): The AlloyDB vectorstore object corresponding to the new collection table.
        destination_table (str): The name of the table to insert the data in.
            Optional. defaults to collection_name.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    if not destination_table:
        warnings.warn(
            f"Destination table not set. Destination table would default to {collection_name}. "
            "Please make sure that there is an existing table with the same name."
        )
        destination_table = collection_name

    # Get row count in PGVector collection
    uuid = await _aget_collection_uuid(engine, collection_name)
    query = f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id='{uuid}'"
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        collection_data_len = result_map.fetchone()
    if collection_data_len is None:
        warnings.warn(f"Collection, {collection_name} contains no elements.")
        return

    # Extract data from the collection and batch insert into the new table
    data_batches = _aextract_pgvector_collection(
        engine, collection_name, batch_size=insert_batch_size
    )

    pending: set[Any] = set()
    max_concurrency = 5
    async for batch_data in data_batches:
        pending.add(
            asyncio.ensure_future(
                vector_store.aadd_embeddings(
                    texts=[data.document for data in batch_data],
                    embeddings=[data.embedding for data in batch_data],
                    metadatas=[data.cmetadata for data in batch_data],
                    ids=[data.id for data in batch_data],
                )
            )
        )
        if len(pending) >= max_concurrency:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
    await asyncio.wait(pending)

    # Validate data migration
    query = f"SELECT COUNT(*) FROM {destination_table}"
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        table_size = result_map.fetchone()
    if not table_size:
        raise ValueError(f"Table: {destination_table} does not exist.")

    if collection_data_len["count"] != table_size["count"]:
        raise ValueError(
            "All data not yet migrated.\n"
            f"Original row count: {collection_data_len['count']}\n"
            f"Collection table, {destination_table} row count: {table_size['count']}"
        )
    elif delete_pg_collection:
        # Delete PGVector data
        query = f"DELETE FROM {EMBEDDINGS_TABLE} WHERE collection_id='{uuid}'"
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

        query = f"DELETE FROM {COLLECTIONS_TABLE} WHERE name='{collection_name}'"
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()
        print(f"Successfully deleted PGVector collection, {collection_name}")


async def _alist_pgvector_collection_names(
    engine: AlloyDBEngine,
) -> List[str]:
    """Lists all collection names present in PGVector table."""
    try:
        query = f"SELECT name from {COLLECTIONS_TABLE}"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            all_rows = result_map.fetchall()
        return [row["name"] for row in all_rows]
    except ProgrammingError as e:
        raise ValueError("Please provide the correct collection table name: " + str(e))


async def aextract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
) -> AsyncIterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.

    Yields:
        The data present in the collection.
    """
    iterator = _aextract_pgvector_collection(engine, collection_name)
    while True:
        try:
            result = await engine._run_as_async(iterator.__anext__())
            yield result
        except StopAsyncIteration:
            break


async def alist_pgvector_collection_names(
    engine: AlloyDBEngine,
) -> List[str]:
    """Lists all collection names present in PGVector table."""
    return await engine._run_as_async(_alist_pgvector_collection_names(engine))


async def amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    destination_table: Optional[str] = None,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (AlloyDBVectorStore): The AlloyDB vectorstore object corresponding to the new collection table.
        destination_table (str): The name of the table to insert the data in.
            Optional. defaults to collection_name.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the AlloyDB table.
            Default: False. Optional.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    await engine._run_as_async(
        _amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            destination_table,
            delete_pg_collection,
            insert_batch_size,
        )
    )


def extract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
) -> Iterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.

    Yields:
        The data present in the collection.
    """
    iterator = _aextract_pgvector_collection(engine, collection_name)
    while True:
        try:
            result = engine._run_as_sync(iterator.__anext__())
            yield result
        except StopAsyncIteration:
            break


def list_pgvector_collection_names(engine: AlloyDBEngine) -> List[str]:
    """Lists all collection names present in PGVector table."""
    return engine._run_as_sync(_alist_pgvector_collection_names(engine))


def migrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    destination_table: Optional[str] = None,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (AlloyDBVectorStore): The AlloyDB vectorstore object corresponding to the new collection table.
        destination_table (str): The name of the table to insert the data in.
            Optional. defaults to collection_name.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    engine._run_as_sync(
        _amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            destination_table,
            delete_pg_collection,
            insert_batch_size,
        )
    )
