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

from typing import Any

from langchain_google_alloydb_pg import AlloyDBEngine


async def acreate_alloydb_client(
    project_id: str,
    region: str,
    cluster: str,
    instance: str,
    db_name: str,
    db_user: str,
    db_pwd: str,
) -> AlloyDBEngine:
    # [START langchain_alloydb_migration_get_client]
    from langchain_google_alloydb_pg import AlloyDBEngine

    engine = await AlloyDBEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster,
        instance=instance,
        database=db_name,
        user=db_user,
        password=db_pwd,
    )
    print("Langchain AlloyDB client initiated.")
    # [END langchain_alloydb_migration_get_client]
    return engine


from langchain_core.embeddings import Embeddings


def get_embeddings_service(
    project_id: str, model_name: str = "textembedding-gecko@001"
) -> Embeddings:
    # [START langchain_alloydb_migration_embedding_service]
    from langchain_google_vertexai import VertexAIEmbeddings

    # choice of model determines the vector size
    embedding_service = VertexAIEmbeddings(project=project_id, model_name=model_name)
    # [END langchain_alloydb_migration_embedding_service]
    print("Langchain Vertex AI Embeddings service initiated.")
    return embedding_service


def get_fake_embeddings_service(vector_size: int = 768) -> Embeddings:
    # [START langchain_alloydb_migration_fake_embedding_service]
    from langchain_core.embeddings import FakeEmbeddings

    embedding_service = FakeEmbeddings(size=vector_size)
    # [END langchain_alloydb_migration_fake_embedding_service]
    print("Langchain Fake Embeddings service initiated.")
    return embedding_service


async def ainit_vector_store(
    engine: AlloyDBEngine,
    table_name: str = "alloydb_table",
    vector_size: int = 768,
    **kwargs: Any,
) -> None:
    # [START langchain_create_alloydb_migration_vector_store_table]
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        **kwargs,
    )
    # [END langchain_create_alloydb_migration_vector_store_table]
    print("Langchain AlloyDB vector store table initialized.")


from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore


async def aget_vector_store(
    engine: AlloyDBEngine,
    embeddings_service: Embeddings,
    table_name: str = "alloydb_table",
    **kwargs: Any,
) -> AlloyDBVectorStore:
    # [START langchain_get_alloydb_migration_vector_store]
    from langchain_google_alloydb_pg import AlloyDBVectorStore

    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        embedding_service=embeddings_service,
        table_name=table_name,
        **kwargs,
    )
    # [END langchain_get_alloydb_migration_vector_store]
    print("Langchain AlloyDB vector store instantiated.")
    return vector_store


async def ainsert_data(
    vector_store: AlloyDBVectorStore,
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> list[str]:
    # [START langchain_alloydb_migration_vector_store_insert_data]
    inserted_ids = await vector_store.aadd_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    # [END langchain_alloydb_migration_vector_store_insert_data]
    print("AlloyDB client inserted the provided data.")
    return inserted_ids
