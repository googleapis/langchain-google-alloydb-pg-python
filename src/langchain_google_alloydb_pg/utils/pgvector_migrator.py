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
import warnings
from typing import List, Optional, Sequence, TypeVar

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import ProgrammingError

from ..engine import AlloyDBEngine

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
) -> Sequence[RowMapping]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        collection_name (str): The name of the collection to get the data for.

    Returns:
        The data present in the collection.
    """
    uuid = await _aget_collection_uuid(engine, collection_name)
    try:
        query = f"SELECT * FROM {EMBEDDINGS_TABLE} WHERE collection_id = '{uuid}'"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch
    except:
        raise ValueError(f"Collection, {collection_name} does not exist.")


async def _ainsert_single_batch(
    engine: AlloyDBEngine,
    destination_table: str,
    metadata_column_names: Optional[List[str]],
    data: Sequence[RowMapping],
    use_json_metadata: Optional[bool] = False,
) -> None:
    """
    Create a batch insert SQL query to insert multiple rows at once.
    Looks like
    INSERT INTO MyTable ( Column1, Column2 )
    VALUES ( Value1, Value2 ), ( Value1, Value2 ), ...

    Args:
        destination_table (str): The name of the table to insert the data in.
        metadata_column_names (str): The metadata columns to be created to keep the data in a row-column format.
            Optional.
        data (Sequence[RowMapping]): All the data (to be inserted) belonging to a pgvector collection.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the AlloyDB table.
            Default: False. Optional.
    """
    params = {}

    if use_json_metadata:
        insert_query = f"INSERT INTO {destination_table} (langchain_id, content, embedding, langchain_metadata) VALUES"

        # Create value clause for the SQL query
        values_clause = ", ".join(
            [
                f"(:langchain_id_{num}, :content_{num}, :embedding_{num}, :langchain_metadata_{num})"
                for num in range(len(data))
            ]
        )
        insert_query += values_clause

        # Add parameters
        for row_number, row in enumerate(data):
            params[f"langchain_id_{row_number}"] = row.id
            params[f"content_{row_number}"] = row.document
            params[f"embedding_{row_number}"] = row.embedding
            params[f"langchain_metadata_{row_number}"] = json.dumps(row.cmetadata)

            row_number += 1

    elif metadata_column_names:
        insert_query = (
            f"INSERT INTO {destination_table} (langchain_id, content, embedding"
        )
        for column in metadata_column_names:
            insert_query += ", " + column
        insert_query += ") VALUES"

        # Create value clause for the SQL query
        values_clause = ", ".join(
            [
                f"(:langchain_id_{num}, :content_{num}, :embedding_{num}, "
                + ", ".join([f":{column}_{num}" for column in metadata_column_names])
                + ")"
                for num in range(len(data))
            ]
        )
        insert_query += values_clause

        # Add parameters
        for row_number, row in enumerate(data):
            params[f"langchain_id_{row_number}"] = row.id
            params[f"content_{row_number}"] = row.document
            params[f"embedding_{row_number}"] = row.embedding
            for column in metadata_column_names:
                # In case the key is not present, add a null value
                if column in row.cmetadata:
                    params[f"{column}_{row_number}"] = row.cmetadata[column]
                else:
                    params[f"{column}_{row_number}"] = None

    # Insert rows
    async with engine._pool.connect() as conn:
        await conn.execute(text(insert_query), params)
        await conn.commit()


async def _arun_all_batch_inserts(
    engine: AlloyDBEngine,
    data: Sequence[RowMapping],
    destination_table: str,
    metadata_column_names: Optional[List[str]],
    use_json_metadata: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Insert all data in batches of 1000 insert queries at once.

    Args:
        data (Sequence[RowMapping]): All the data (to be inserted) belonging to a PGVector collection.
        destination_table (str): The name of the table to insert the data in.
        metadata_column_names (str): The metadata columns to be created to keep the data in a row-column format.
            Optional.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the AlloyDB table.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    data_size = len(data)
    for i in range(data_size // insert_batch_size):
        await _ainsert_single_batch(
            engine,
            destination_table,
            metadata_column_names,
            data=data[insert_batch_size * i : insert_batch_size * (i + 1)],
            use_json_metadata=use_json_metadata,
        )
    if data_size % insert_batch_size:
        i = data_size // insert_batch_size
        await _ainsert_single_batch(
            engine,
            destination_table,
            metadata_column_names,
            data=data[
                insert_batch_size * i : insert_batch_size * i
                + data_size % insert_batch_size
            ],
            use_json_metadata=use_json_metadata,
        )
    print("All rows inserted successfully.")


async def _amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
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
        collection_name (str): The collection to migrate.
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

    collection_data = await _aextract_pgvector_collection(engine, collection_name)

    await _arun_all_batch_inserts(
        engine,
        data=collection_data,
        destination_table=destination_table,
        metadata_column_names=(
            [column for column in metadata_columns] if metadata_columns else None
        ),
        insert_batch_size=insert_batch_size,
        use_json_metadata=use_json_metadata,
    )

    # Validate all data migration and delete old data
    if delete_pg_collection:
        query = f"SELECT COUNT(*) FROM {destination_table}"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            table_size = result_map.fetchone()
        if not table_size:
            raise ValueError(f"Table: {destination_table} does not exist.")
        if len(collection_data) != table_size["count"]:
            raise ValueError(
                "All data not yet migrated. The pre-existing data would not be deleted."
            )
        uuid = await _aget_collection_uuid(engine, collection_name)

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
) -> Sequence[RowMapping]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        collection_name (str): The name of the collection to get the data for.

    Returns:
        The data present in the collection.
    """
    return await engine._run_as_async(
        _aextract_pgvector_collection(engine, collection_name)
    )


async def alist_pgvector_collection_names(
    engine: AlloyDBEngine,
) -> List[str]:
    """Lists all collection names present in PGVector table."""
    return await engine._run_as_async(_alist_pgvector_collection_names(engine))


async def amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
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
        collection_name (str): The collection to migrate.
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
) -> Sequence[RowMapping]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        collection_name (str): The name of the collection to get the data for.

    Returns:
        The data present in the collection.
    """
    return engine._run_as_sync(_aextract_pgvector_collection(engine, collection_name))


def list_pgvector_collection_names(engine: AlloyDBEngine) -> List[str]:
    """Lists all collection names present in PGVector table."""
    return engine._run_as_sync(_alist_pgvector_collection_names(engine))


def migrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
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
        collection_name (str): The collection to migrate.
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
            metadata_columns,
            destination_table,
            use_json_metadata,
            delete_pg_collection,
            insert_batch_size,
        )
    )
