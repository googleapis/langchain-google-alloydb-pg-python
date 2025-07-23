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
import uuid

import sqlalchemy
from langchain_community.document_loaders import CSVLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import Column

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

EMBEDDING_COUNT = 100000
VECTOR_SIZE = 768


PROJECT_ID = ""

# AlloyDB info
REGION = ""
CLUSTER_NAME = ""
INSTANCE_NAME = ""
DATABASE_NAME = ""
USER = ""  # Use your super user `postgres`
PASSWORD = ""

# AlloyDB Omni info for running ScaNN index only
OMNI_HOST = ""
OMNI_DATABASE_NAME = ""
OMNI_USER = ""  # Use your super user `postgres`
OMNI_PASSWORD = ""

vector_table_name = "wine_reviews_vector"

# Dataset
DATASET_PATH = "wine_reviews_dataset.csv"
dataset_columns = [
    "country",
    "description",
    "designation",
    "points",
    "price",
    "province",
    "region_1",
    "region_2",
    "taster_name",
    "taster_twitter_handle",
    "title",
    "variety",
    "winery",
]


def load_csv_documents(dataset_path=DATASET_PATH):
    """Loads documents directly from a CSV file using LangChain."""

    loader = CSVLoader(file_path=dataset_path)
    documents = loader.load()
    return documents[0:EMBEDDING_COUNT]


async def get_engine(is_omni=False):
    if is_omni:
        connstring = f"postgresql+asyncpg://{OMNI_USER}:{OMNI_PASSWORD}@{OMNI_HOST}:5432/{OMNI_DATABASE_NAME}"
        print(f"Connecting to AlloyDB Omni with {connstring}")

        async_engine = sqlalchemy.ext.asyncio.create_async_engine(connstring)
        engine = AlloyDBEngine.from_engine(async_engine)
        print("Successfully connected to AlloyDB Omni database.")
    else:
        engine = await AlloyDBEngine.afrom_instance(
            project_id=PROJECT_ID,
            region=REGION,
            cluster=CLUSTER_NAME,
            instance=INSTANCE_NAME,
            database=DATABASE_NAME,
            user=USER,
            password=PASSWORD,
        )
        print("Successfully connected to AlloyDB database.")
    return engine


async def create_vector_store_table(documents, engine):
    print("Initializaing Vectorstore tables...")
    await engine.ainit_vectorstore_table(
        table_name=vector_table_name,
        vector_size=VECTOR_SIZE,
        metadata_columns=[
            Column("country", "VARCHAR", nullable=True),
            Column("description", "VARCHAR", nullable=True),
            Column("designation", "VARCHAR", nullable=True),
            Column("points", "VARCHAR", nullable=True),
            Column("price", "INTEGER", nullable=True),
            Column("province", "VARCHAR", nullable=True),
            Column("region_1", "VARCHAR", nullable=True),
            Column("region_2", "VARCHAR", nullable=True),
            Column("taster_name", "VARCHAR", nullable=True),
            Column("taster_twitter_handle", "VARCHAR", nullable=True),
            Column("title", "VARCHAR", nullable=True),
            Column("variety", "VARCHAR", nullable=True),
            Column("winery", "VARCHAR", nullable=True),
        ],
        overwrite_existing=True,  # Enabling this will recreate the table if exists.
    )
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )

    # Initialize AlloyDBVectorStore
    print("Initializing VectorStore...")
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
        metadata_columns=dataset_columns,
    )

    ids = [str(uuid.uuid4()) for i in range(len(documents))]
    await vector_store.aadd_documents(documents, ids)
    print("Vector table created.")


async def main():
    parser = argparse.ArgumentParser(description="Your script's description")
    parser.add_argument(
        "--omni", action="store_true", help="Running on AlloyDB Omni instance,"
    )
    args = parser.parse_args()
    if args.omni:
        engine = await get_engine(is_omni=True)
    else:
        engine = await get_engine()
    documents = load_csv_documents()
    await create_vector_store_table(documents, engine)
    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
