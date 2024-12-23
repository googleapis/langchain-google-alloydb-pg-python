#!/usr/bin/env python

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
from typing import Any, Iterator

"""Migrate PineconeVectorStore to Langchain AlloyDBVectorStore.

Given a pinecone index, the following code fetches the data from pinecone
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
PINECONE_API_KEY = "YOUR_API_KEY"
PINECONE_INDEX_NAME = "YOUR_INDEX_NAME"
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "YOUR_REGION"
CLUSTER = "YOUR_CLUSTER_ID"
INSTANCE = "YOUR_INSTANCE_ID"
DB_NAME = "YOUR_DATABASE_ID"
DB_USER = "YOUR_DATABASE_USERNAME"
DB_PWD = "YOUR_DATABASE_PASSWORD"

# TODO(developer): Optional, change the values below.
PINECONE_NAMESPACE = ""
PINECONE_VECTOR_SIZE = 768
PINECONE_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"

from pinecone import Index  # type: ignore


def get_ids_batch(
    pinecone_index: Index,
    pinecone_namespace: str = PINECONE_NAMESPACE,
    pinecone_batch_size: int = PINECONE_BATCH_SIZE,
) -> Iterator[list[str]]:
    results = pinecone_index.list_paginated(
        prefix="", namespace=pinecone_namespace, limit=pinecone_batch_size
    )
    ids = [v.id for v in results.vectors]
    yield ids

    while results.pagination is not None:
        pagination_token = results.pagination.next
        results = pinecone_index.list_paginated(
            prefix="", pagination_token=pagination_token, limit=pinecone_batch_size
        )

        # Extract and yield the next batch of IDs
        ids = [v.id for v in results.vectors]
        yield ids
    # [END pinecone_get_ids_batch]
    print("Pinecone client fetched all ids from index.")


def get_data_batch(
    pinecone_index: Index, id_iterator: Iterator[list[str]]
) -> Iterator[tuple[list[str], list[Any], list[str], list[Any]]]:
    # [START pinecone_get_data_batch]
    # Iterate through the IDs and download their contents
    for ids in id_iterator:
        # Fetch vectors for the current batch of IDs
        all_data = pinecone_index.fetch(ids=ids)
        ids = []
        embeddings = []
        contents = []
        metadatas = []

        # Process each vector in the current batch
        for doc in all_data["vectors"].values():
            ids.append(doc["id"])
            embeddings.append(doc["values"])
            contents.append(str(doc["metadata"]))
            metadata = doc["metadata"]
            metadatas.append(metadata)

        # Yield the current batch of results
        yield ids, embeddings, contents, metadatas
    # [END pinecone_get_data_batch]
    print("Pinecone client fetched all data from index.")


async def main(
    pinecone_api_key: str = PINECONE_API_KEY,
    pinecone_index_name: str = PINECONE_INDEX_NAME,
    pinecone_namespace: str = PINECONE_NAMESPACE,
    pinecone_vector_size: int = PINECONE_VECTOR_SIZE,
    pinecone_batch_size: int = PINECONE_BATCH_SIZE,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    cluster: str = CLUSTER,
    instance: str = INSTANCE,
    alloydb_table: str = ALLOYDB_TABLE_NAME,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_pwd: str = DB_PWD,
) -> None:
    # [START pinecone_get_client]
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pinecone_client = Pinecone(
        api_key=pinecone_api_key,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # [END pinecone_get_client]
    print("Pinecone client initiated.")

    # [START pinecone_get_index]
    pinecone_index = pinecone_client.Index(pinecone_index_name)
    # [END pinecone_get_index]
    print("Pinecone index reference initiated.")

    from alloydb_snippets import acreate_alloydb_client

    alloydb_engine = await acreate_alloydb_client(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        db_name=db_name,
        db_user=db_user,
        db_pwd=db_pwd,
    )

    # [START pinecone_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store, get_embeddings_service

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=pinecone_vector_size,
        overwrite_existing=True,
    )

    embeddings_service = get_embeddings_service(
        project_id, model_name=EMBEDDING_MODEL_NAME
    )
    vs = await aget_vector_store(
        engine=alloydb_engine,
        embeddings_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END pinecone_alloydb_migration_get_alloydb_vectorstore]
    print("Pinecone migration AlloyDBVectorStore table created.")

    id_iterator = get_ids_batch(pinecone_index, pinecone_namespace, pinecone_batch_size)
    for ids, embeddings, contents, metadatas in get_data_batch(
        pinecone_index=pinecone_index,
        id_iterator=id_iterator,
    ):
        # [START pinecone_alloydb_migration_insert_data_batch]
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        # [END pinecone_alloydb_migration_insert_data_batch]
    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
