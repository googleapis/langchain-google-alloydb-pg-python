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

"""Migrate MilvusVectorStore to Langchain AlloyDBVectorStore.
Given a milvus collection, the following code fetches the data from milvus
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
PROJECT_ID = "dishaprakash-playground"
REGION = "us-central1"
CLUSTER = "my-cluster-2"
INSTANCE = "my-cluster-2-primary"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PWD = "demo-project"

# TODO(developer): Optional, change the values below.
MILVUS_URI = ""
MILVUS_COLLECTION_NAME = ""
MILVUS_VECTOR_SIZE = 768
MILVUS_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"

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
        output_fields=["pk", "source", "location", "text", "vector"],
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

    from alloydb_snippets import acreate_alloydb_client, get_embeddings_service

    alloydb_engine = await acreate_alloydb_client(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        db_name=db_name,
        db_user=db_user,
        db_pwd=db_pwd,
    )

    embeddings_service = get_embeddings_service(
        project_id, model_name=EMBEDDING_MODEL_NAME
    )

    # [START milvus_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=milvus_vector_size,
        overwrite_existing=True,
    )

    vs = await aget_vector_store(
        engine=alloydb_engine,
        embeddings_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END milvus_alloydb_migration_get_alloydb_vectorstore]
    print("Milvus migration AlloyDBVectorStore table created.")

    data_iterator = get_data_batch(
        milvus_client=milvus_client,
        milvus_batch_size=milvus_batch_size,
        milvus_collection_name=milvus_collection_name,
    )

    for ids, contents, embeddings, metadatas in data_iterator:
        # [START milvus_alloydb_migration_insert_data_batch]
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        # [END milvus_alloydb_migration_insert_data_batch]

    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
