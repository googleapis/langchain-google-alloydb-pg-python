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

import os
from typing import Any, Optional

# TODO(dev): Replace the values below
project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster = os.environ["CLUSTER_ID"]
instance = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]

# TODO(dev): (optional values) Replace the values below
db_user = os.environ.get("DB_USER", "")
db_pwd = os.environ.get("DB_PASSWORD", "")
table_name = os.environ.get("TABLE_NAME", "alloy_db_migration_table")
vector_size = int(os.environ.get("VECTOR_SIZE", "768"))


# [START langchain_alloydb_migration_get_client]
from langchain_google_alloydb_pg import AlloyDBEngine


async def aget_alloydb_client(
    project_id: str = project_id,
    region: str = region,
    cluster: str = cluster,
    instance: str = instance,
    database: str = db_name,
    user: Optional[str] = db_user,
    password: Optional[str] = db_pwd,
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


# [END langchain_alloydb_migration_get_client]

# [START langchain_alloydb_migration_fake_embedding_service]
from langchain_core.embeddings import FakeEmbeddings


def get_embeddings_service(size: int = vector_size) -> FakeEmbeddings:
    embeddings_service = FakeEmbeddings(size=size)

    print("Langchain FakeEmbeddings service initiated.")
    return embeddings_service


# [END langchain_alloydb_migration_fake_embedding_service]


# [START langchain_create_alloydb_migration_vector_store_table]
async def ainit_vector_store(
    engine: AlloyDBEngine,
    table_name: str = table_name,
    vector_size: int = vector_size,
    **kwargs: Any,
) -> None:
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        **kwargs,
    )

    print("Langchain AlloyDB vector store table initialized.")


# [END langchain_create_alloydb_migration_vector_store_table]


# [START langchain_get_alloydb_migration_vector_store]
from langchain_core.embeddings import Embeddings

from langchain_google_alloydb_pg import AlloyDBVectorStore


async def aget_vector_store(
    engine: AlloyDBEngine,
    embeddings_service: Embeddings,
    table_name: str = table_name,
    **kwargs: Any,
) -> AlloyDBVectorStore:
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        embedding_service=embeddings_service,
        table_name=table_name,
        **kwargs,
    )

    print("Langchain AlloyDB vector store instantiated.")
    return vector_store


# [END langchain_get_alloydb_migration_vector_store]


# [START langchain_alloydb_migration_vector_store_insert_data]
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

    print("AlloyDB client inserted the provided data.")
    return inserted_ids


# [END langchain_alloydb_migration_vector_store_insert_data]
