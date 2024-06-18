import asyncio
import time
import numpy as np
from google.cloud.alloydb.connector import AsyncConnector, IPTypes

EMBEDDING_COUNT = 100000

# AlloyDB info
PROJECT_ID = "starter-akitsch"
REGION = "us-central1"  # @param {type:"string"}
CLUSTER_NAME = "my-alloy-db"  # @param {type:"string"}
INSTANCE_NAME = "my-primary"  # @param {type:"string"}
DATABASE_NAME = "langchain"  # @param {type:"string"}
USER = "postgres"  # @param {type:"string"}
PASSWORD = "my-pg-password"  # @param {type:"string"}


from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBVectorStore,
    Column,
)
from langchain_google_alloydb_pg.indexes import (
    HNSWIndex,
    IVFFlatIndex,
    DistanceStrategy,
)
from langchain_google_vertexai import VertexAIEmbeddings

DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN
k = 10
query_1 = "Brooding aromas of barrel spice"
query_2 = "Aromas include tropical fruit, broom, brimstone and dried herb."
query = query_2
query_vectors = [query_1, query_2]

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)


async def get_vector_store():
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
        distance_strategy=DISTANCE_STRATEGY,
        table_name="wines_vector",
        embedding_service=embedding,
    )
    return vector_store


async def hnsw_search(vector_store):
    print("------------------------HNSW------------------------")
    # Distance strategy: EUCLIDEAN, COSINE_DISTANCE, INNER_PRODUCT
    hnsw_index = HNSWIndex(
        name="hnsw", distance_strategy=DISTANCE_STRATEGY, m=99, ef_construction=200
    )
    await vector_store.aapply_vector_index(hnsw_index)
    assert await vector_store.is_valid_index(hnsw_index.name)

    start_time = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end_time = time.monotonic()  # timer ends
    latency = end_time - start_time

    await vector_store.adrop_vector_index(hnsw_index.name)
    return docs, latency


async def ivfflat_search(vector_store):
    print("------------------------IVFFLAT------------------------")
    ivfflat_index = IVFFlatIndex(name="ivfflat", distance_strategy=DISTANCE_STRATEGY)
    await vector_store.aapply_vector_index(ivfflat_index)
    assert await vector_store.is_valid_index(ivfflat_index.name)

    start = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end = time.monotonic()  # timer ends

    await vector_store.adrop_vector_index(ivfflat_index.name)
    latency = round(end - start, 2)
    return docs, latency


async def knn_search(vector_store):
    print("------------------------KNN------------------------")
    start = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end = time.monotonic()  # timer ends

    latency = round(end - start, 2)
    return docs, latency


def calculate_recall(base, target) -> float:
    # size of intersection / total number of times
    base = {doc.metadata["id"] for doc in base}
    target = {doc.metadata["id"] for doc in target}
    return len(base & target) / len(base)


async def main():
    vector_store = await get_vector_store()
    knn_docs, knn_latency = await knn_search(vector_store)
    hnsw_docs, hnsw_latency = await hnsw_search(vector_store)
    ivfflat_docs, ivfflat_latency = await ivfflat_search(vector_store)
    # hnsw_recall = calculate_recall(knn_docs, hnsw_docs)
    # ivfflat_recall = calculate_recall(knn_docs, ivfflat_docs)

    # print(f"KNN recall: 1.0            KNN latency: {knn_latency}")
    # print(f"HNSW recall: {hnsw_recall}          HNSW latency: {hnsw_latency}")
    # print(f"IVFFLAT recall: {ivfflat_recall}    IVFFLAT latency: {ivfflat_latency}")


if __name__ == "__main__":
    asyncio.run(main())
