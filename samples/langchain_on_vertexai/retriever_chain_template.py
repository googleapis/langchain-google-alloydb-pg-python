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
from typing import Any

import vertexai
from config import (
    CLUSTER,
    DATABASE,
    INSTANCE,
    PASSWORD,
    PROJECT_ID,
    REGION,
    STAGING_BUCKET,
    TABLE_NAME,
    USER,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from vertexai import agent_engines

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

# This sample requires a vector store table
# Create these tables using `AlloyDBEngine` method `init_vectorstore_table()`
# or create and load the table using `create_embeddings.py`


engine = None  # Use global variable to share connection pooling


def runnable_builder(model: ChatVertexAI, **kwargs: Any) -> Runnable:
    """Builds a runnable.
    Args:
        model: The LLM model to use.
    Returns:
        A Runnable object.
    """
    global engine
    if not engine:  # Reuse connection pool
        engine = AlloyDBEngine.from_instance(
            PROJECT_ID,
            REGION,
            CLUSTER,
            INSTANCE,
            DATABASE,
            user=USER,
            password=PASSWORD,
        )

    # Create a retriever tool
    vector_store = AlloyDBVectorStore.create_sync(
        engine,
        table_name=TABLE_NAME,
        embedding_service=VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=PROJECT_ID
        ),
    )
    retriever = vector_store.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# Initialize VertexAI
client = vertexai.Client(
    project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET
)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "AlloyDBRetriever"

agent = agent_engines.LangchainAgent(
    model="gemini-2.0-flash-001",
    runnable_builder=runnable_builder,
)
remote_app = client.agent_engines.create(
    agent=agent,
    config={
        "display_name": "AlloyDBRetriever",
        "sys_version": "3.11",
        "requirements": [
            "langchain_google_alloydb_pg",
            "langchain",
            "langchain-google-vertexai",
            "google-cloud-aiplatform",
        ],
        "extra_packages": ["config.py"],
    },
)
print(remote_app.query(input={"input": "movies about engineers"})["answer"])
