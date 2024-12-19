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
import sys
import uuid
from typing import Any, Optional

# [START langchain_alloydb_get_client]
from langchain_google_alloydb_pg import AlloyDBEngine


async def aget_client(
    project_id: str,
    region: str,
    cluster: str,
    instance: str,
    database: str,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> AlloyDBEngine:
    engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        database=database,
        user=user,
        password=password,
    )

    print("Langchain AlloyDB client initiated.")
    return engine


# [END langchain_alloydb_get_client]

# [START langchain_alloydb_fake_embedding_service]
from langchain_core.embeddings import FakeEmbeddings


def get_embeddings_service(size: int) -> FakeEmbeddings:
    embeddings_service = FakeEmbeddings(size=size)

    print("Langchain FakeEmbeddings service initiated.")
    return embeddings_service


# [END langchain_alloydb_fake_embedding_service]


# [START langchain_create_alloydb_vector_store_table]
async def ainit_vector_store(
    engine: AlloyDBEngine, table_name: str, vector_size: int, **kwargs: Any
) -> None:
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        overwrite_existing=True,
        **kwargs,
    )

    print("Langchain AlloyDB vector store table initialized.")


# [END langchain_create_alloydb_vector_store_table]


# [START langchain_get_alloydb_vector_store]
from langchain_core.embeddings import Embeddings

from langchain_google_alloydb_pg import AlloyDBVectorStore


async def aget_vector_store(
    engine: AlloyDBEngine, embeddings_service: Embeddings, table_name: str
) -> AlloyDBVectorStore:
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        embedding_service=embeddings_service,
        table_name=table_name,
    )

    print("Langchain AlloyDB vector store instantiated.")
    return vector_store


# [END langchain_get_alloydb_vector_store]


# [START langchain_alloydb_vector_store_insert_data]
async def ainsert_data(
    vector_store: AlloyDBVectorStore,
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> list[str]:
    inserted_ids = await vector_store.aadd_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print("AlloyDB client fetched all data from index.")
    return inserted_ids


# [END langchain_alloydb_vector_store_insert_data]


async def main() -> None:
    client = await aget_client(
        project_id=sys.argv[1],
        region=sys.argv[2],
        cluster=sys.argv[3],
        instance=sys.argv[4],
        database=sys.argv[5],
        user=sys.argv[6],
        password=sys.argv[7],
    )
    # In case you're using a different embeddings service, choose one from [LangChain's Embedding models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/).
    embeddings_service = get_embeddings_service(size=768)
    await ainit_vector_store(
        engine=client,
        table_name=sys.argv[8],
        vector_size=768,
    )
    vs = await aget_vector_store(
        engine=client,
        embeddings_service=embeddings_service,
        table_name=sys.argv[8],
    )
    # sample rows
    ids = [str(uuid.uuid4())]
    texts = ["content_1"]
    embeddings = embeddings_service.embed_documents(texts)
    metadatas: list[dict[str, Any]] = [{} for _ in texts]
    ids = await ainsert_data(
        vector_store=vs,
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    await client.close()
    print(f"Inserted {len(ids)} values to Langchain Alloy DB Vector Store.")


if __name__ == "__main__":
    asyncio.run(main())
