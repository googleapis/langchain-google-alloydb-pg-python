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

"""Migrate WeaviateVectorStore to Langchain AlloyDBVectorStore.
Given a weaviate collection, the following code fetches the data from weaviate
in batches and uploads to an AlloyDBVectorStore.
"""

# TODO(dev): Replace the values below
WEAVIATE_API_KEY = "YOUR_API_KEY"
WEAVIATE_CLUSTER_URL = "YOUR_CLUSTER_URL"
EMBEDDING_API_KEY = "YOUR_EMBEDDING_API_KEY"
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "YOUR_REGION"
CLUSTER = "YOUR_CLUSTER_ID"
INSTANCE = "YOUR_INSTANCE_ID"
DB_NAME = "YOUR_DATABASE_ID"
DB_USER = "YOUR_DATABASE_USERNAME"
DB_PWD = "YOUR_DATABASE_PASSWORD"

# TODO(developer): Optional, change the values below.
WEAVIATE_COLLECTION_NAME = ""
WEAVIATE_VECTOR_SIZE = 768
WEAVIATE_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"

from weaviate.collections import Collection  # type: ignore


def get_data_batch(
    weaviate_collection: Collection, weaviate_batch_size: int = WEAVIATE_BATCH_SIZE
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START weaviate_get_data_batch]
    # Iterate through the IDs and download their contents

    ids = []
    content = []
    embeddings = []
    metadatas = []

    for item in weaviate_collection.iterator(include_vector=True):
        ids.append(str(item.uuid))
        content.append(item.properties["page_content"])
        embeddings.append(item.vector["default"])
        metadatas.append(item.properties["metadata"])

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
    weaviate_cluster_url: str = WEAVIATE_CLUSTER_URL,
    weaviate_vector_size: int = WEAVIATE_VECTOR_SIZE,
    weaviate_batch_size: int = WEAVIATE_BATCH_SIZE,
    embedding_api_key: str = EMBEDDING_API_KEY,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    cluster: str = CLUSTER,
    instance: str = INSTANCE,
    alloydb_table: str = ALLOYDB_TABLE_NAME,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_pwd: str = DB_PWD,
) -> None:
    # [START weaviate_get_client]
    import weaviate

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        headers={"X-Cohere-Api-Key": embedding_api_key},
    )

    # [END weaviate_get_client]
    print("Weaviate client initiated.")

    # [START weaviate_get_collection]
    weaviate_collection = weaviate_client.collections.get(weaviate_collection_name)
    # [END weaviate_get_collection]
    print("Weaviate collection reference initiated.")

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

    # [START weaviate_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store, get_embeddings_service

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=weaviate_vector_size,
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
    # [END weaviate_alloydb_migration_get_alloydb_vectorstore]
    print("Weaviate migration AlloyDBVectorStore table created.")

    data_iterator = get_data_batch(
        weaviate_collection=weaviate_collection, weaviate_batch_size=weaviate_batch_size
    )

    for ids, contents, embeddings, metadatas in data_iterator:
        # [START weaviate_alloydb_migration_insert_data_batch]
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        # [END weaviate_alloydb_migration_insert_data_batch]

    weaviate_client.close()
    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
