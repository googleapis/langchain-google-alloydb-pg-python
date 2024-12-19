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
import os
from typing import Any, Iterator

"""Migrate Pinecone to Langchain AlloyDBVectorStore.

Given a pinecone index, the following code fetches the data from pinecone
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
pinecone_api_key = os.environ["PINECONE_API_KEY"]

# TODO(dev): (optional values) Replace the values below
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "sample-movies")
pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "")
pinecone_serverless_cloud = os.environ.get("PINECONE_SERVERLESS_CLOUD", "aws")
pinecone_serverless_region = os.environ.get("PINECONE_SERVERLESS_REGION", "us-east-1")
pinecone_migration_table = os.environ.get(
    "PINECONE_MIGRATION_TABLE", "pinecone_migration"
)
pinecone_batch_size = int(os.environ.get("PINECONE_BATCH_SIZE", "100"))
pinecone_vector_size = int(os.environ.get("PINECONE_VECTOR_SIZE", "1024"))

# [START pinecone_get_ids_batch]
from pinecone import Index  # type: ignore


def get_ids_batch(
    pinecone_index: Index, namespace: str = "", batch_size: int = 100
) -> Iterator[list[str]]:
    results = pinecone_index.list_paginated(
        prefix="", namespace=namespace, limit=batch_size
    )
    ids = [v.id for v in results.vectors]
    yield ids

    while results.pagination is not None:
        pagination_token = results.pagination.next
        results = pinecone_index.list_paginated(
            prefix="", pagination_token=pagination_token, limit=batch_size
        )

        # Extract and yield the next batch of IDs
        ids = [v.id for v in results.vectors]
        yield ids
    print("Pinecone client fetched all ids from index.")


# [END pinecone_get_ids_batch]


# [START pinecone_get_data_batch]
from pinecone import Index  # type: ignore


def get_data_batch(
    pinecone_index: Index, namespace: str = "", batch_size: int = 100
) -> Iterator[tuple[list[str], list[Any], list[str], list[Any]]]:

    id_iterator = get_ids_batch(pinecone_index, namespace, batch_size)
    # Iterate through the batches of IDs and process them
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
    print("Pinecone client fetched all data from index.")


# [END pinecone_get_data_batch]


async def main() -> None:
    # [START pinecone_get_client]
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pinecone_client = Pinecone(
        api_key=pinecone_api_key,
        spec=ServerlessSpec(
            cloud=pinecone_serverless_cloud, region=pinecone_serverless_region
        ),
    )
    print("Pinecone client initiated.")
    # [END pinecone_get_client]

    # [START pinecone_get_index]
    pinecone_index = pinecone_client.Index(pinecone_index_name)
    print("Pinecone index reference initiated.")
    # [END pinecone_get_index]

    from alloydb_snippets import aget_alloydb_client

    alloydb_engine = await aget_alloydb_client()

    # [START pinecone_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store, get_embeddings_service

    from langchain_google_alloydb_pg import Column

    # Note that the vector size and id_column name/type are configurable.
    # We need to customize the vector store table because the sample data has
    # 1024 vectors and integer like id values (not UUIDs).
    await alloydb_engine.ainit_vectorstore_table(
        table_name=pinecone_migration_table,
        vector_size=pinecone_vector_size,
        id_column=Column("langchain_id", "text", nullable=False),
        overwrite_existing=True,
    )
    print("Pinecone migration AlloyDBVectorStore table created.")

    embeddings_service = get_embeddings_service(pinecone_vector_size)
    vs = await aget_vector_store(
        engine=alloydb_engine,
        embeddings_service=embeddings_service,
        table_name=pinecone_migration_table,
    )
    # [END pinecone_alloydb_migration_get_alloydb_vectorstore]

    # [START pinecone_alloydb_migration_insert_data_batch]
    for ids, embeddings, contents, metadatas in get_data_batch(
        pinecone_index=pinecone_index,
        namespace=pinecone_namespace,
        batch_size=pinecone_batch_size,
    ):
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    print("Migration completed, inserted all the batches of data to AlloyDB.")
    # [END pinecone_alloydb_migration_insert_data_batch]


if __name__ == "__main__":
    asyncio.run(main())
