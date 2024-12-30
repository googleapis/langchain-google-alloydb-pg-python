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

"""Migrate QdrantVectorStore to Langchain AlloyDBVectorStore.
Given a qdrant collection, the following code fetches the data from qdrant
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
QDRANT_COLLECTION_NAME = ""
QDRANT_PATH = ""
QDRANT_VECTOR_SIZE = 768
QDRANT_BATCH_SIZE = 10
ALLOYDB_TABLE_NAME = "alloydb_table"
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"

from qdrant_client import QdrantClient  # type: ignore


def get_data_batch(
    qdrant_client: QdrantClient,
    qdrant_batch_size: int = QDRANT_BATCH_SIZE,
    qdrant_collection_name: str = QDRANT_COLLECTION_NAME,
) -> Iterator[tuple[list[str], list[Any], list[list[float]], list[Any]]]:
    # [START qdrant_get_data_batch]
    # Iterate through the IDs and download their contents
    offset = None
    while True:
        docs, offset = qdrant_client.scroll(
            collection_name=qdrant_collection_name,
            with_vectors=True,
            limit=qdrant_batch_size,
            offset=offset,
            with_payload=True,
        )

        ids: List[str] = []
        contents: List[Any] = []
        embeddings: List[List[float]] = []
        metadatas: List[Any] = []

        for doc in docs:
            if doc.payload and doc.vector:
                ids.append(str(doc.id))
                contents.append(doc.payload["page_content"])
                embeddings.append(doc.vector)  # type: ignore
                metadatas.append(doc.payload["metadata"])

        yield ids, contents, embeddings, metadatas

        if not offset:
            break

    # [END qdrant_get_data_batch]
    print("Qdrant client fetched all data from collection.")


async def main(
    qdrant_collection_name: str = QDRANT_COLLECTION_NAME,
    qdrant_vector_size: int = QDRANT_VECTOR_SIZE,
    qdrant_batch_size: int = QDRANT_BATCH_SIZE,
    qdrant_path: str = QDRANT_PATH,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    cluster: str = CLUSTER,
    instance: str = INSTANCE,
    alloydb_table: str = ALLOYDB_TABLE_NAME,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_pwd: str = DB_PWD,
) -> None:
    # [START qdrant_get_client]
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(path=qdrant_path)

    # [END qdrant_get_client]
    print("Qdrant client initiated.")

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

    # [START qdrant_alloydb_migration_get_alloydb_vectorstore]
    from alloydb_snippets import aget_vector_store

    await alloydb_engine.ainit_vectorstore_table(
        table_name=alloydb_table,
        vector_size=qdrant_vector_size,
        overwrite_existing=True,
    )

    vs = await aget_vector_store(
        engine=alloydb_engine,
        embeddings_service=embeddings_service,
        table_name=alloydb_table,
    )
    # [END qdrant_alloydb_migration_get_alloydb_vectorstore]
    print("Qdrant migration AlloyDBVectorStore table created.")

    data_iterator = get_data_batch(
        qdrant_client=qdrant_client,
        qdrant_batch_size=qdrant_batch_size,
        qdrant_collection_name=qdrant_collection_name,
    )

    for ids, contents, embeddings, metadatas in data_iterator:
        # [START qdrant_alloydb_migration_insert_data_batch]
        inserted_ids = await vs.aadd_embeddings(
            texts=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        # [END qdrant_alloydb_migration_insert_data_batch]

    print("Migration completed, inserted all the batches of data to AlloyDB.")


if __name__ == "__main__":
    asyncio.run(main())
