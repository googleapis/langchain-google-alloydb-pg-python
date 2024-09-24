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
import json
import warnings
from typing import AsyncIterator, Iterator, List, Optional, Sequence, TypeVar

from langchain_core.embeddings import Embeddings
from sqlalchemy import RowMapping, text
from sqlalchemy.exc import ProgrammingError

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_google_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore

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
) -> AsyncIterator[RowMapping]:
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
                row = result_proxy.fetchone()
                if not row:
                    break
                yield row._mapping
    except:
        raise ValueError(f"Collection, {collection_name} does not exist.")


async def _ainsert_single_batch(
    engine: AlloyDBEngine,
    vector_store: AsyncAlloyDBVectorStore,
    data: Sequence[RowMapping],
) -> None:
    """
    Create a batch insert SQL query to insert multiple rows at once.
    Looks like
    INSERT INTO MyTable ( Column1, Column2 )
    VALUES ( Value1, Value2 ), ( Value1, Value2 ), ...

    Args:
        vector_store (AlloyDBVectorStore): The AlloyDB vector store corresponding to the table.
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        data (Sequence[RowMapping]): All the data (to be inserted) belonging to a pgvector collection.
    """
    metadata_col_names = (
        ", " + ", ".join(vector_store.metadata_columns)
        if len(vector_store.metadata_columns) > 0
        else ""
    )
    insert_stmt = f'INSERT INTO "{vector_store.schema_name}"."{vector_store.table_name}"({vector_store.id_column}, {vector_store.content_column}, {vector_store.embedding_column}{metadata_col_names}'
    insert_stmt += (
        f", {vector_store.metadata_json_column})"
        if vector_store.metadata_json_column
        else ")"
    )

    values_stmt = "VALUES (:id, :content, :embedding"
    for metadata_column in vector_store.metadata_columns:
        values_stmt += f", :{metadata_column}"
    if vector_store.metadata_json_column:
        values_stmt += f", :extra_metadata"
    values_stmt += ")"

    params = []
    for row in data:
        curr_value = {
            "id": row.id,
            "content": row.document,
            "embedding": row.embedding,
        }
        metadata = row.cmetadata
        extra_metadata = row.cmetadata
        for metadata_column in vector_store.metadata_columns:
            curr_value[metadata_column] = (
                metadata[metadata_column] if metadata_column in metadata else None
            )
            del extra_metadata[metadata_column]
        if vector_store.metadata_json_column:
            curr_value["extra_metadata"] = (
                json.dumps(extra_metadata) if extra_metadata else None
            )
        params.append(curr_value)

    # Insert rows
    insert_query = insert_stmt + values_stmt
    async with engine._pool.connect() as conn:
        await conn.execute(text(insert_query), params)
        await conn.commit()


async def _batch_data(generator, batch_size):
    """
    Batches data from a generator.

    Args:
        generator: A generator function yielding data items.
        batch_size: The desired batch size.

    Yields:
        Batches of data as lists.
    """
    batch = []
    async for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:  # Yield remaining items if any
        yield batch


async def _amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    embeddings_service: Embeddings,
    metadata_columns: Optional[List[str]] = [],
    destination_table: Optional[str] = None,
    use_json_metadata: Optional[bool] = False,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        embeddings_service (Embeddings): The embeddings service used in the vectorstore.
            Embeddings are directly copied from the PGVector database. This service is used only to initialise the vector store.
        metadata_columns (List[str]): The metadata columns to be created to keep the data in a row-column format.
            Optional.
        destination_table (str): The name of the table to insert the data in.
            Optional. defaults to collection_name.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the AlloyDB table.
            Default: False. Optional.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    if not use_json_metadata and not metadata_columns:
        raise ValueError(
            "Please specify the columns for the new table. "
            "To store data in JSON format, set use_json_metadata=True when calling this method."
        )

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
    collection_data = _aextract_pgvector_collection(engine, collection_name)
    data_batches = _batch_data(collection_data, insert_batch_size)

    if metadata_columns:
        vector_store = await AsyncAlloyDBVectorStore.create(
            engine=engine,
            embedding_service=embeddings_service,
            table_name=destination_table,
            metadata_columns=metadata_columns,
        )
    else:
        vector_store = await AsyncAlloyDBVectorStore.create(
            engine=engine,
            embedding_service=embeddings_service,
            table_name=destination_table,
        )
    tasks = [
        asyncio.create_task(
            _ainsert_single_batch(
                engine,
                vector_store,
                data=batch_data,
            )
        )
        async for batch_data in data_batches
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Validate data migration
    query = f"SELECT COUNT(*) FROM {destination_table}"
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        table_size = result_map.fetchone()
    if not table_size:
        raise ValueError(f"Table: {destination_table} does not exist.")

    if collection_data_len["count"] != table_size["count"]:
        raise ValueError("All data not yet migrated.")
    elif delete_pg_collection:
        # Delete PGVector data
        query = f"DELETE FROM {COLLECTIONS_TABLE} WHERE name='{collection_name}'"
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

        query = f"DELETE FROM {EMBEDDINGS_TABLE} WHERE collection_id='{uuid}'"
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
) -> AsyncIterator[RowMapping]:
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
    embeddings_service: Embeddings,
    metadata_columns: Optional[List[str]] = [],
    destination_table: Optional[str] = None,
    use_json_metadata: Optional[bool] = False,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        embeddings_service (Embeddings): The embeddings service used in the vectorstore.
            Embeddings are directly copied from the PGVector database. This service is used only to initialise the vector store.
        metadata_columns (List[str]): The metadata columns to be created to keep the data in a row-column format.
            Optional.
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
            embeddings_service,
            metadata_columns,
            destination_table,
            use_json_metadata,
            delete_pg_collection,
            insert_batch_size,
        )
    )


def extract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
) -> Iterator[RowMapping]:
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
    embeddings_service: Embeddings,
    metadata_columns: Optional[List[str]] = [],
    destination_table: Optional[str] = None,
    use_json_metadata: Optional[bool] = False,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the AlloyDB interface.

    Args:
        engine (AlloyDBEngine): The AlloyDB engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        embeddings_service (Embeddings): The embeddings service used in the vectorstore.
            Embeddings are directly copied from the PGVector database. This service is used only to initialise the vector store.
        metadata_columns (List[str]): The metadata columns to be created to keep the data in a row-column format.
            Optional.
        destination_table (str): The name of the table to insert the data in.
            Optional. defaults to collection_name.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the AlloyDB table.
            Default: False. Optional.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    engine._run_as_sync(
        _amigrate_pgvector_collection(
            engine,
            collection_name,
            embeddings_service,
            metadata_columns,
            destination_table,
            use_json_metadata,
            delete_pg_collection,
            insert_batch_size,
        )
    )
