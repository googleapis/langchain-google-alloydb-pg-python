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

"""Migrate Chroma to LangChain AlloyDBVectorStore.
Given a Chroma collection, the following code fetches the data from Chroma
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
CHROMADB_PATH = "./chroma_langchain_db"
CHROMADB_COLLECTION_NAME = "example_collection"
PROJECT_ID = "my-project-id"
REGION = "us-central1"
CLUSTER = "my-cluster"
INSTANCE = "my-instance"
DB_NAME = "my-db"
DB_USER = "postgres"
DB_PWD = "secret-password"

# TODO(developer): Optional, change the values below.
VECTOR_SIZE = 768
CHROMADB_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
MAX_CONCURRENCY = 100

from langchain_chroma import Chroma  # type: ignore


def get_data_batch(
    chromadb_client: Chroma, chromadb_batch_size: int = CHROMADB_BATCH_SIZE
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START chromadb_get_data_batch]
    # Iterate through the IDs and download their contents
    offset = 0
    while True:
        docs = chromadb_client.get(
            include=["metadatas", "documents", "embeddings"],
            limit=chromadb_batch_size,
            offset=offset,
        )

        if len(docs["documents"]) == 0:
            break

        yield docs["ids"], docs["documents"], docs["embeddings"].tolist(), docs[
            "metadatas"
        ]

        offset += chromadb_batch_size

    # [END chromadb_get_data_batch]
    print("ChromaDB client fetched all data from collection.")


async def main(
    chromadb_collection_name: str = CHROMADB_COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
    chromadb_batch_size: int = CHROMADB_BATCH_SIZE,
    chromadb_path: str = CHROMADB_PATH,
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
    # [START chromadb_vectorstore_alloydb_migration_embedding_service]
    # The VectorStore interface requires an embedding service. This workflow does not
    # generate new embeddings, therefore FakeEmbeddings class is used to avoid any costs.
    from langchain_core.embeddings import FakeEmbeddings

    embeddings_service = FakeEmbeddings(size=vector_size)
    # [END chromadb_vectorstore_alloydb_migration_embedding_service]
    print("Langchain Fake Embeddings service initiated.")

    # [START chromadb_get_client]
    from langchain_chroma import Chroma

    chromadb_client = Chroma(
        collection_name=chromadb_collection_name,
        embedding_function=embeddings_service,
        persist_directory=chromadb_path,
    )
    # [END chromadb_get_client]
    print("ChromaDB vectorstore reference initiated.")

    # [START chromadb_vectorstore_alloydb_migration_get_client]
    from langchain_google_alloydb_pg import AlloyDBEngine

    alloydb_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        database=db_name,
        user=db_user,
        password=db_pwd,
        ip_type=IPTypes.PUBLIC,
    )
    # [END chromadb_vectorstore_alloydb_migration_get_client]
    print("Langchain AlloyDB client initiated.")

    # [START chromadb_vectorstore_alloydb_migration_create_table]
    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=vector_size,
    )
    # [END chromadb_vectorstore_alloydb_migration_create_table]
    print("Langchain AlloyDB vectorstore table created.")

    # [START chromadb_vectorstore_alloydb_migration_vector_store]
    from langchain_google_alloydb_pg import AlloyDBVectorStore

    vs = await AlloyDBVectorStore.create(
        engine=alloydb_engine,
        embedding_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END chromadb_vectorstore_alloydb_migration_vector_store]
    print("Langchain AlloyDBVectorStore initialized.")

    data_iterator = get_data_batch(
        chromadb_client=chromadb_client,
        chromadb_batch_size=chromadb_batch_size,
    )

    # [START chromadb_vectorstore_alloydb_migration_insert_data_batch]
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
    # [END chromadb_vectorstore_alloydb_migration_insert_data_batch]
    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
