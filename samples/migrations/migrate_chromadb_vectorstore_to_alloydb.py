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

"""Migrate ChromaDBVectorStore to Langchain AlloyDBVectorStore.
Given a chromadb collection, the following code fetches the data from chromadb
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "YOUR_REGION"
CLUSTER = "YOUR_CLUSTER_ID"
INSTANCE = "YOUR_INSTANCE_ID"
DB_NAME = "YOUR_DATABASE_ID"
DB_USER = "YOUR_DATABASE_USERNAME"
DB_PWD = "YOUR_DATABASE_PASSWORD"

# TODO(developer): Optional, change the values below.
CHROMADB_COLLECTION_NAME = ""
CHROMADB_PATH = ""
CHROMADB_VECTOR_SIZE = 768
CHROMADB_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"

from langchain_chroma import Chroma  # type: ignore


def get_data_batch(
    chromadb_vectorstore: Chroma, chromadb_batch_size: int = CHROMADB_BATCH_SIZE
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START chromadb_get_data_batch]
    # Iterate through the IDs and download their contents
    offset = 0
    while True:
        docs = chromadb_vectorstore.get(
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
    chromadb_vector_size: int = CHROMADB_VECTOR_SIZE,
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
) -> None:
    from alloydb_snippets import get_embeddings_service

    embeddings_service = get_embeddings_service(
        project_id, model_name=EMBEDDING_MODEL_NAME
    )
    # [START chromadb_get_vectorstore]
    from langchain_chroma import Chroma

    chromadb_vector_store = Chroma(
        collection_name=chromadb_collection_name,
        embedding_function=embeddings_service,
        persist_directory=chromadb_path,
    )

    # [END chromadb_get_vectorstore]

    print("ChromaDB vectorstore reference initiated.")

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

    # [START chromadb_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=chromadb_vector_size,
        overwrite_existing=True,
    )

    vs = await aget_vector_store(
        engine=alloydb_engine,
        embeddings_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END chromadb_alloydb_migration_get_alloydb_vectorstore]
    print("ChromaDB migration AlloyDBVectorStore table created.")

    data_iterator = get_data_batch(
        chromadb_vectorstore=chromadb_vector_store,
        chromadb_batch_size=chromadb_batch_size,
    )

    for ids, contents, embeddings, metadatas in data_iterator:
        # [START chromadb_alloydb_migration_insert_data_batch]
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        # [END chromadb_alloydb_migration_insert_data_batch]

    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
