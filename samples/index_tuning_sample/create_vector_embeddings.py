import asyncio
import uuid

from google.cloud.alloydb.connector import AsyncConnector, IPTypes
from langchain.docstore.document import Document
from langchain_community.document_loaders import CSVLoader
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine

EMBEDDING_COUNT = 100

# AlloyDB info
PROJECT_ID = "duwenxin-space"
REGION = "us-central1"  # @param {type:"string"}
CLUSTER_NAME = "my-alloydb-cluster"  # @param {type:"string"}
INSTANCE_NAME = "my-alloydb-instance"  # @param {type:"string"}
DATABASE_NAME = "netflix"  # @param {type:"string"}
USER = "postgres"  # @param {type:"string"}
PASSWORD = "postgres"  # @param {type:"string"}

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

connection_string = f"projects/{PROJECT_ID}/locations/{REGION}/clusters/{CLUSTER_NAME}/instances/{INSTANCE_NAME}"
# initialize Connector object
connector = AsyncConnector()


async def getconn():
    conn = await connector.connect(
        connection_string,
        "asyncpg",
        user=USER,
        password=PASSWORD,
        db=DATABASE_NAME,
        enable_iam_auth=False,
        ip_type=IPTypes.PUBLIC,
    )
    return conn


# create connection pool
pool = create_async_engine(
    "postgresql+asyncpg://", async_creator=getconn, isolation_level="AUTOCOMMIT"
)


async def load_csv_documents(dataset_path=DATASET_PATH):
    """Loads documents directly from a CSV file using LangChain."""

    loader = CSVLoader(file_path=dataset_path)
    documents = loader.load()

    documents = [
        Document(page_content=str(doc.dict()), metadata=doc.metadata)
        for doc in documents
    ]

    return documents


async def create_vector_store_table(documents):
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
    print("Initializaing Vectorstore tables...")
    await engine.ainit_vectorstore_table(
        table_name=vector_table_name,
        vector_size=768,
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

    ids = [str(uuid.uuid4()) for i in range(EMBEDDING_COUNT)]
    await vector_store.aadd_documents(documents, ids)
    print("Vector table created.")


async def main():
    documents = await load_csv_documents()
    await create_vector_store_table(documents)


if __name__ == "__main__":
    asyncio.run(main())
