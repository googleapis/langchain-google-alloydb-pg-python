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
    CHAT_TABLE_NAME,
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from vertexai import agent_engines

from langchain_google_alloydb_pg import (
    AlloyDBChatMessageHistory,
    AlloyDBEngine,
    AlloyDBVectorStore,
)

# This sample requires a vector store table and a chat message table
# Create these tables using `AlloyDBEngine` methods
# `init_vectorstore_table()` and `init_chat_history_table()`
# or create and load the tables using `create_embeddings.py`

engine = None  # Use global variable to share connection pooling


def get_session_history(session_id: str) -> AlloyDBChatMessageHistory:
    """fetches chat history from an AlloyDB table
    Args:
        session_id: The user's session id.
    Returns:
        A AlloyDBChatMessageHistory object.
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
    return AlloyDBChatMessageHistory.create_sync(
        engine=engine, session_id=session_id, table_name=CHAT_TABLE_NAME
    )


def runnable_builder(model: ChatVertexAI, **kwargs: Any) -> RunnableWithMessageHistory:
    """Builds a runnable with message history.
    Args:
        model: The LLM model to use.
    Returns:
        A RunnableWithMessageHistory object.
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
    tool = create_retriever_tool(
        retriever, "search_movies", "Searches and returns movies."
    )
    tools = [tool]

    # Create the agent
    instructions = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
    )
    base_prompt = ChatPromptTemplate.from_template(instructions)
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    # Create a runnable that manages chat message history for the agent
    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# Initialize VertexAI
client = vertexai.Client(
    project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET
)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "AlloyDBAgent"

agent = agent_engines.LangchainAgent(
    model="gemini-2.0-flash-001",
    runnable_builder=runnable_builder,
)
remote_app = client.agent_engines.create(
    agent=agent,
    config={
        "display_name": "AlloyDBAgent",
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

print(
    remote_app.query(
        input="What movies are about engineers?",
        config={"configurable": {"session_id": "abc123"}},
    )["output"]
)
