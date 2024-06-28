# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import vertexai  # type: ignore
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

# This sample requires a vector store table
# Create these tables using `AlloyDBEngine`` method: init_vectorstore_table()

# Replace the following variables with your values
PROJECT_ID = os.getenv("PROJECT_ID") or "my-project-id"
STAGING_BUCKET = os.getenv("STAGING_BUCKET") or "gs://my-bucket"
REGION = os.getenv("REGION") or "us-central1"
CLUSTER = os.getenv("CLUSTER") or "my-alloy-db"
INSTANCE = os.getenv("INSTANCE") or "my-primary"
DATABASE = os.getenv("DATABASE") or "my_database"
TABLE_NAME = os.getenv("TABLE_NAME") or "my_test_table"
USER = os.getenv("DB_USER") or "postgres"
PASSWORD = os.getenv("DB_PASSWORD") or "password"


def similarity_search(query: str):
    """Searches and returns movies.

    Args:
      query: The user query to search for related items

    Returns:
      List[Document]: A list of Documents
    """
    engine = AlloyDBEngine.from_instance(
        PROJECT_ID,
        REGION,
        CLUSTER,
        INSTANCE,
        DATABASE,
        # To use IAM authentication, remove user and password and ensure
        # the Reasoning Engine Agent service account is a database user
        # with access to the vector store table
        user="postgres",
        password=PASSWORD,
    )

    vector_store = AlloyDBVectorStore.create_sync(
        engine,
        table_name=TABLE_NAME,
        embedding_service=VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=PROJECT_ID
        ),
    )
    retriever = vector_store.as_retriever()
    return retriever.invoke(query)


# Uncomment to test locally

# app = reasoning_engines.LangchainAgent(
#     model="gemini-1.0-pro",
#     tools=[similarity_search],
#     model_kwargs={
#         "temperature": 0.1,
#     },
# )
# app.set_up()
# print(app.query(input="movies about engineers"))

# Initialize VertexAI
vertexai.init(project=PROJECT_ID, location="us-central1", staging_bucket=STAGING_BUCKET)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "PrebuiltAgent"

remote_app = reasoning_engines.ReasoningEngine.create(
    reasoning_engines.LangchainAgent(
        model="gemini-1.0-pro",
        tools=[similarity_search],
        model_kwargs={
            "temperature": 0.1,
        },
    ),
    requirements="requirements.txt",
    display_name="PrebuiltAgent",
    sys_version="3.11",
)

print(remote_app.query(input="movies about engineers"))
