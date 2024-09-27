# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import time

import sqlalchemy
from create_vector_embeddings import (
    CLUSTER_NAME,
    DATABASE_NAME,
    INSTANCE_NAME,
    OMNI_DATABASE_NAME,
    OMNI_HOST,
    OMNI_PASSWORD,
    OMNI_USER,
    PASSWORD,
    PROJECT_ID,
    REGION,
    USER,
    vector_table_name,
)
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import (
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFIndex,
    ScaNNIndex,
)

k = 10
query_1 = "Brooding aromas of barrel spice."
query_2 = "Aromas include tropical fruit, broom, brimstone and dried herb."
query_3 = "Wine from spain."
query_4 = "Condensed and dark on the bouquet"
query_5 = "Light, fresh and silky—just what might be expected from cool-climate Pinot Noir"
queries = [query_1, query_2, query_3, query_4, query_5]


embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)


async def get_vector_store():
    """Get vector store instance."""
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=CLUSTER_NAME,
        instance=INSTANCE_NAME,
        database=DATABASE_NAME,
        user=USER,
        password=PASSWORD,
    )

    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
        index_query_options=HNSWQueryOptions(ef_search=256),
    )
    return vector_store


async def get_omni_vector_store():
    connstring = f"postgresql+asyncpg://{OMNI_USER}:{OMNI_PASSWORD}@{OMNI_HOST}:5432/{OMNI_DATABASE_NAME}"
    print(f"Connecting to AlloyDB Omni with {connstring}")

    async_engine = sqlalchemy.ext.asyncio.create_async_engine(connstring)
    omni_engine = AlloyDBEngine.from_engine(async_engine)
    omni_vector_store = await AlloyDBVectorStore.create(
        omni_engine,
        embedding_service=embedding,
        table_name=vector_table_name,
    )
    return omni_vector_store


async def query_vector_with_timing(vector_store, query):
    """Query using the vector with timing"""
    start_time = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(k=k, query=query)
    end_time = time.monotonic()  # timer ends
    latency = end_time - start_time
    return docs, latency


async def hnsw_search(vector_store, knn_docs):
    """Create an HNSW index and perform similaity search with the index."""
    hnsw_index = HNSWIndex(name="hnsw", m=36, ef_construction=96)
    await vector_store.aapply_vector_index(hnsw_index)
    assert await vector_store.ais_valid_index(hnsw_index.name)
    print("HNSW index created.")
    latencies = []
    recalls = []

    for i in range(len(queries)):
        hnsw_docs, latency = await query_vector_with_timing(
            vector_store, queries[i]
        )
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], hnsw_docs))

    await vector_store.adrop_vector_index(hnsw_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall


async def ivfflat_search(vector_store, knn_docs):
    """Create an IVFFlat index and perform similaity search with the index."""
    ivfflat_index = IVFFlatIndex(name="ivfflat")
    await vector_store.aapply_vector_index(ivfflat_index)
    assert await vector_store.ais_valid_index(ivfflat_index.name)
    print("IVFFLAT index created.")
    latencies = []
    recalls = []

    for i in range(len(queries)):
        ivfflat_docs, latency = await query_vector_with_timing(
            vector_store, queries[i]
        )
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], ivfflat_docs))

    await vector_store.adrop_vector_index(ivfflat_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall


async def ivf_search(vector_store, knn_docs):
    """Create an IVF index and perform similaity search with the index."""
    ivf_index = IVFIndex(name="ivf")
    await vector_store.aapply_vector_index(ivf_index)
    assert await vector_store.ais_valid_index(ivf_index.name)
    print("IVF index created.")
    latencies = []
    recalls = []

    for i in range(len(queries)):
        ivf_docs, latency = await query_vector_with_timing(
            vector_store, queries[i]
        )
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], ivf_docs))

    await vector_store.adrop_vector_index(ivf_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall


async def scann_search(vector_store, knn_docs):
    """Create an ScaNN index and perform similaity search with the index."""
    scann_index = ScaNNIndex(name="scann")
    await vector_store.aapply_vector_index(scann_index)
    assert await vector_store.ais_valid_index(scann_index.name)
    print("ScaNN index created.")
    latencies = []
    recalls = []

    for i in range(len(queries)):
        scann_docs, latency = await query_vector_with_timing(
            vector_store, queries[i]
        )
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], scann_docs))

    await vector_store.adrop_vector_index(scann_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall


async def knn_search(vector_store):
    """Perform similaity search without index."""
    latencies = []
    knn_docs = []
    for query in queries:
        docs, latency = await query_vector_with_timing(vector_store, query)
        latencies.append(latency)
        knn_docs.append(docs)
    average_latency = sum(latencies) / len(latencies)
    return knn_docs, average_latency


def calculate_recall(base, target):
    """Calculate recall on the target result."""
    # size of intersection / total number of times
    base = {doc.page_content for doc in base}
    target = {doc.page_content for doc in target}
    return len(base & target) / len(base)


async def main():
    parser = argparse.ArgumentParser(description="Your script's description")
    parser.add_argument(
        "--omni", action="store_true", help="Running on AlloyDB Omni instance,"
    )
    args = parser.parse_args()
    if args.omni:
        # Running ScaNN index benchmark on AlloyDB Omni.
        vector_store = await get_omni_vector_store()
        knn_docs, knn_latency = await knn_search(vector_store)
        scann_average_latency, scann_average_recall = await scann_search(
            vector_store, knn_docs
        )
        print(f"KNN recall: 1.0            KNN latency: {knn_latency}")
        print(
            f"ScaNN average recall: {scann_average_recall}         ScaNN average latency: {scann_average_latency}"
        )
    else:
        # Running HNSW, IVFFlat, IVF indexes benchmark on Cloud AlloyDB.
        vector_store = await get_vector_store()
        knn_docs, knn_latency = await knn_search(vector_store)
        hnsw_average_latency, hnsw_average_recall = await hnsw_search(
            vector_store, knn_docs
        )
        ivfflat_average_latency, ivfflat_average_recall = await ivfflat_search(
            vector_store, knn_docs
        )
        ivf_average_latency, ivf_average_recall = await ivf_search(
            vector_store, knn_docs
        )

        print(f"KNN recall: 1.0               KNN latency: {knn_latency}")
        print(
            f"HNSW average recall: {hnsw_average_recall}      HNSW average latency: {hnsw_average_latency}"
        )
        print(
            f"IVFFLAT average recall: {ivfflat_average_recall}   IVFFLAT latency: {ivfflat_average_latency}"
        )
        print(
            f"IVF average recall: {ivf_average_recall}       IVF latency: {ivf_average_latency}"
        )
    await vector_store._engine.close()
    await vector_store._engine._connector.close()


if __name__ == "__main__":
    asyncio.run(main())
