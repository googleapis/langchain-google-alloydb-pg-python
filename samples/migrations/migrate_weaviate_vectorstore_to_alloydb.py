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

"""Migrate WeaviateVectorStore to Langchain AlloyDBVectorStore.
Given a weaviate collection, the following code fetches the data from weaviate
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
WEAVIATE_API_KEY = "my-wv-api-key"
WEAVIATE_CLUSTER_URL = "my-wv-cluster-url"
WEAVIATE_COLLECTION_NAME = "example_collection"
PROJECT_ID = "my-project-id"
REGION = "us-central1"
CLUSTER = "my-cluster"
INSTANCE = "my-instance"
DB_NAME = "my-db"
DB_USER = "postgres"
DB_PWD = "secret-password"

# TODO(developer): Optional, change the values below.
WEAVIATE_TEXT_KEY = "text"
VECTOR_SIZE = 768
WEAVIATE_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
MAX_CONCURRENCY = 100

from weaviate import WeaviateClient


def get_data_batch(
    weaviate_client: WeaviateClient,
    weaviate_collection_name: str = WEAVIATE_COLLECTION_NAME,
    weaviate_text_key: str = WEAVIATE_TEXT_KEY,
    weaviate_batch_size: int = WEAVIATE_BATCH_SIZE,
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START weaviate_get_data_batch]
    # Iterate through the IDs and download their contents
    weaviate_collection = weaviate_client.collections.get(weaviate_collection_name)
    ids = []
    content = []
    embeddings = []
    metadatas = []

    for item in weaviate_collection.iterator(include_vector=True):
        ids.append(str(item.uuid))
        content.append(item.properties[weaviate_text_key])
        embeddings.append(item.vector["default"])
        del item.properties[weaviate_text_key]  # type: ignore
        metadatas.append(item.properties)

        if len(ids) >= weaviate_batch_size:
            # Yield the current batch of results
            yield ids, content, embeddings, metadatas
            # Reset lists to start a new batch
            ids = []
            content = []
            embeddings = []
            metadatas = []
    # [END weaviate_get_data_batch]
    print("Weaviate client fetched all data from collection.")


async def main(
    weaviate_api_key: str = WEAVIATE_API_KEY,
    weaviate_collection_name: str = WEAVIATE_COLLECTION_NAME,
    weaviate_text_key: str = WEAVIATE_TEXT_KEY,
    weaviate_cluster_url: str = WEAVIATE_CLUSTER_URL,
    vector_size: int = VECTOR_SIZE,
    weaviate_batch_size: int = WEAVIATE_BATCH_SIZE,
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
    # [START weaviate_get_client]
    import weaviate

    # For a locally running weaviate instance, use `weaviate.connect_to_local()`
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
    )
    # [END weaviate_get_client]
    print("Weaviate client initiated.")

    # [START weaviate_vectorstore_alloydb_migration_embedding_service]
    # The VectorStore interface requires an embedding service. This workflow does not
    # generate new embeddings, therefore FakeEmbeddings class is used to avoid any costs.
    from langchain_core.embeddings import FakeEmbeddings

    embeddings_service = FakeEmbeddings(size=vector_size)
    # [END weaviate_vectorstore_alloydb_migration_embedding_service]
    print("Langchain Fake Embeddings service initiated.")

    # [START weaviate_vectorstore_alloydb_migration_get_client]
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
    # [END weaviate_vectorstore_alloydb_migration_get_client]
    print("Langchain AlloyDB client initiated.")

    # [START weaviate_vectorstore_alloydb_migration_create_table]
    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=vector_size,
    )

    # [END weaviate_vectorstore_alloydb_migration_create_table]
    print("Langchain AlloyDB vectorstore table created.")

    # [START weaviate_vectorstore_alloydb_migration_vector_store]
    from langchain_google_alloydb_pg import AlloyDBVectorStore

    vs = await AlloyDBVectorStore.create(
        engine=alloydb_engine,
        embedding_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END weaviate_vectorstore_alloydb_migration_vector_store]
    print("Langchain AlloyDBVectorStore initialized.")

    data_iterator = get_data_batch(
        weaviate_client=weaviate_client,
        weaviate_collection_name=weaviate_collection_name,
        weaviate_text_key=weaviate_text_key,
        weaviate_batch_size=weaviate_batch_size,
    )
    # [START weaviate_vectorstore_alloydb_migration_insert_data_batch]
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
    # [END weaviate_vectorstore_alloydb_migration_insert_data_batch]
    print("Migration completed, inserted all the batches of data to AlloyDB.")
    weaviate_client.close()


if __name__ == "__main__":
    asyncio.run(main())
