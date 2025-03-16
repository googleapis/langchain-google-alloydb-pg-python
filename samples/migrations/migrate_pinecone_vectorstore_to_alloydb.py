#!/usr/bin/env python

# Copyright 2025 Google LLC
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
from typing import Any, Iterator

from google.cloud.alloydb.connector import IPTypes

"""Migrate PineconeVectorStore to Langchain AlloyDBVectorStore.

Given a pinecone index, the following code fetches the data from pinecone
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
PINECONE_API_KEY = "my-pc-api-key"
PINECONE_INDEX_NAME = "my-pc-index-name"
PROJECT_ID = "my-project-id"
REGION = "us-central1"
CLUSTER = "my-cluster"
INSTANCE = "my-instance"
DB_NAME = "my-db"
DB_USER = "postgres"
DB_PWD = "secret-password"
PINECONE_CONTENT_COLUMN_NAME = 'title'  # Define the metadata key that will be moved to the content column in the AlloyDB vector store

# TODO(developer): Optional, change the values below.
PINECONE_NAMESPACE = ""
VECTOR_SIZE = 768
PINECONE_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
MAX_CONCURRENCY = 100

from pinecone import Index  # type: ignore


def get_ids_batch(
    pinecone_index: Index,
    pinecone_namespace: str = PINECONE_NAMESPACE,
    pinecone_batch_size: int = PINECONE_BATCH_SIZE,
) -> Iterator[list[str]]:
    """
    Fetches IDs from a Pinecone index in batches, handling pagination correctly.
    Uses a generator to yield batches of IDs.
    """
    # [START pinecone_get_ids_batch]
    results = pinecone_index.list_paginated(
        prefix="", namespace=pinecone_namespace, limit=pinecone_batch_size
    )
    ids = [v.id for v in results.vectors]
    if ids:  # Prevents yielding an empty list.
        yield ids

    # Check BOTH pagination and pagination.next
    while results.pagination is not None and results.pagination.get("next") is not None:
        pagination_token = results.pagination.get("next")
        results = pinecone_index.list_paginated(
            prefix="",
            pagination_token=pagination_token,
            namespace=pinecone_namespace,
            limit=pinecone_batch_size,
        )

        # Extract and yield the next batch of IDs
        ids = [v.id for v in results.vectors]
        if ids:  # Prevents yielding an empty list.
            yield ids
    # [END pinecone_get_ids_batch]
    print("Pinecone client fetched all ids from index.")


def get_data_batch(
    pinecone_index: Index, 
    pinecone_namespace: str, 
    pinecone_batch_size: int,
    pinecone_content_column_name: str = PINECONE_CONTENT_COLUMN_NAME
) -> Iterator[tuple[list[str], list[str], list[Any], list[Any]]]:
    id_iterator = get_ids_batch(pinecone_index, pinecone_namespace, pinecone_batch_size)
    # [START pinecone_get_data_batch]
    import uuid

    # Iterate through the IDs and download their contents
    for ids in id_iterator:
        all_data = pinecone_index.fetch(ids=ids, namespace=pinecone_namespace)
        ids = []
        embeddings = []
        contents = []
        metadatas = []

        # Process each vector in the current batch
        for doc in all_data["vectors"].values():
            # You might need to update this data translation logic according to one or more of your field names
            # id is the unqiue identifier for the content
            if "id" in doc:
                ids.append(doc["id"])
            else:
                # Generate a uuid if id column is missing in source
                ids.append(str(uuid.uuid4()))
            # values is the vector embedding of the content
            embeddings.append(doc["values"])
            # Check if pinecone_content_column_name exists in metadata before accessing
            if pinecone_content_column_name in doc.metadata:
                contents.append(str(doc.metadata[pinecone_content_column_name]))
                del doc.metadata[pinecone_content_column_name]  # Remove pinecone_content_column_name after processing
            else:
                contents.append("")  # Or handle the missing pinecone_content_column_name field appropriately
            # metadata is the additional context
            metadata = doc["metadata"]
            metadatas.append(metadata)

        # Yield the current batch of results
        yield ids, contents, embeddings, metadatas
    # [END pinecone_get_data_batch]
    print("Pinecone client fetched all data from index.")


async def main(
    pinecone_api_key: str = PINECONE_API_KEY,
    pinecone_index_name: str = PINECONE_INDEX_NAME,
    pinecone_namespace: str = PINECONE_NAMESPACE,
    vector_size: int = VECTOR_SIZE,
    pinecone_batch_size: int = PINECONE_BATCH_SIZE,
    pinecone_content_column_name: str = PINECONE_CONTENT_COLUMN_NAME,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    cluster: str = CLUSTER,
    instance: str = INSTANCE,
    alloydb_table: str = ALLOYDB_TABLE_NAME,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_pwd: str = DB_PWD,
    max_concurrency: int = MAX_CONCURRENCY,
) -> None:
    # [START pinecone_get_client]
    from pinecone import Pinecone  # type: ignore

    pinecone_client = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pinecone_client.Index(pinecone_index_name)
    # [END pinecone_get_client]
    print("Pinecone index reference initiated.")

    # [START pinecone_vectorstore_alloydb_migration_get_client]
    from langchain_google_alloydb_pg import AlloyDBEngine

    alloydb_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        database=db_name,
        user=db_user,
        password=db_pwd,
        ip_type=IPTypes.PUBLIC,  # Optionally use IPTypes.PRIVATE
    )
    # [END pinecone_vectorstore_alloydb_migration_get_client]
    print("Langchain AlloyDB client initiated.")

    # [START pinecone_vectorstore_alloydb_migration_create_table]
    from langchain_google_alloydb_pg import Column

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=vector_size,
        # Customize the ID column types if not using the UUID data type
        # id_column=Column("langchain_id", "TEXT") #  Default is Column("langchain_id", "UUID")
        # overwrite_existing=True, # Drop the old table and Create a new vector store table
    )
    # [END pinecone_vectorstore_alloydb_migration_create_table]
    print("Langchain AlloyDB vectorstore table created.")

    # [START pinecone_vectorstore_alloydb_migration_embedding_service]
    # The VectorStore interface requires an embedding service. This workflow does not
    # generate new embeddings, therefore FakeEmbeddings class is used to avoid any costs.
    from langchain_core.embeddings import FakeEmbeddings

    embeddings_service = FakeEmbeddings(size=vector_size)
    # [END pinecone_vectorstore_alloydb_migration_embedding_service]
    print("Langchain Fake Embeddings service initiated.")

    # [START pinecone_vectorstore_alloydb_migration_vector_store]
    from langchain_google_alloydb_pg import AlloyDBVectorStore

    vs = await AlloyDBVectorStore.create(
        engine=alloydb_engine,
        embedding_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END pinecone_vectorstore_alloydb_migration_vector_store]
    print("Langchain AlloyDBVectorStore initialized.")

    data_iterator = get_data_batch(
        pinecone_index, pinecone_namespace, pinecone_batch_size, pinecone_content_column_name
    )

    # [START pinecone_vectorstore_alloydb_migration_insert_data_batch]
    pending: set[Any] = set()
    for ids, contents, embeddings, metadatas in data_iterator:
        pending.add(
            asyncio.ensure_future(
                vs.aadd_embeddings(
                    texts=contents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
            )
        )
        if len(pending) >= max_concurrency:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
    if pending:
        await asyncio.wait(pending)
    # [END pinecone_vectorstore_alloydb_migration_insert_data_batch]
    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
