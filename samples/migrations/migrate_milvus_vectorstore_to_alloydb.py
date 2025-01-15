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
from typing import Any, Iterator, List

"""Migrate Milvus to Langchain AlloyDBVectorStore.
Given a Milvus collection, the following code fetches the data from Milvus
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

# TODO(developer): Change the values below.
MILVUS_URI = "./milvus_data"
MILVUS_COLLECTION_NAME = "test_milvus"
MILVUS_VECTOR_SIZE = 768
MILVUS_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"
MAX_CONCURRENCY = 100

from pymilvus import MilvusClient  # type: ignore


def get_data_batch(
    milvus_client: MilvusClient,
    milvus_batch_size: int = MILVUS_BATCH_SIZE,
    milvus_collection_name: str = MILVUS_COLLECTION_NAME,
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START milvus_get_data_batch]
    # Iterate through the IDs and download their contents
    iterator = milvus_client.query_iterator(
        collection_name=milvus_collection_name,
        filter='pk >= "0"',
        output_fields=["pk", "text", "vector", "idv"],
        batch_size=milvus_batch_size,
    )

    while True:
        ids = []
        content = []
        embeddings = []
        metadatas = []
        page = iterator.next()
        if len(page) == 0:
            iterator.close()
            break
        for i in range(len(page)):
            doc = page[i]
            ids.append(doc["pk"])
            content.append(doc["text"])
            embeddings.append(doc["vector"])
            del doc["pk"]
            del doc["text"]
            del doc["vector"]
            metadatas.append(doc)

        yield ids, content, embeddings, metadatas

    # [END milvus_get_data_batch]
    print("Milvus client fetched all data from collection.")


async def main(
    milvus_collection_name: str = MILVUS_COLLECTION_NAME,
    milvus_vector_size: int = MILVUS_VECTOR_SIZE,
    milvus_batch_size: int = MILVUS_BATCH_SIZE,
    milvus_uri: str = MILVUS_URI,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    cluster: str = CLUSTER,
    instance: str = INSTANCE,
    alloydb_table: str = ALLOYDB_TABLE_NAME,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_pwd: str = DB_PWD,
) -> None:
    # [START milvus_get_client]
    milvus_client = MilvusClient(uri=milvus_uri)

    # [END milvus_get_client]
    print("Milvus client initiated.")

    # [START langchain_alloydb_migration_get_client]
    from langchain_google_alloydb_pg import AlloyDBEngine

    alloydb_engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        database=db_name,
        user=db_user,
        password=db_pwd,
    )
    # [END langchain_alloydb_migration_get_client]
    print("Langchain AlloyDB client initiated.")

    # [START langchain_alloydb_migration_fake_embedding_service]
    from langchain_core.embeddings import FakeEmbeddings

    embeddings_service = FakeEmbeddings(size=milvus_vector_size)
    # [END langchain_alloydb_migration_fake_embedding_service]
    print("Langchain Fake Embeddings service initiated.")

    # [START milvus_migration_alloydb_vectorstore]

    # [START langchain_create_alloydb_migration_vector_store_table]

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=milvus_vector_size,
        overwrite_existing=True,
    )

    # [END langchain_create_alloydb_migration_vector_store_table]
    print("Langchain AlloyDB vector store table initialized.")

    # [START langchain_get_alloydb_migration_vector_store]

    from langchain_google_alloydb_pg import AlloyDBVectorStore

    vs = await AlloyDBVectorStore.create(
        engine=alloydb_engine,
        embedding_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END langchain_get_alloydb_migration_vector_store]
    print("Langchain AlloyDB vector store instantiated.")

    # [END milvus_migration_alloydb_vectorstore]
    print("Milvus migration AlloyDBVectorStore table created.")

    data_iterator = get_data_batch(
        milvus_client=milvus_client,
        milvus_batch_size=milvus_batch_size,
        milvus_collection_name=milvus_collection_name,
    )

    # [START milvus_alloydb_migration_insert_data_batch]

    # [START langchain_alloydb_migration_vector_store_insert_data]
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
        if len(pending) >= MAX_CONCURRENCY:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
    if pending:
        await asyncio.wait(pending)
    # [END langchain_alloydb_migration_vector_store_insert_data]

    # [END milvus_alloydb_migration_insert_data_batch]

    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
