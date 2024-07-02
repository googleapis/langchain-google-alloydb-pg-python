# LangChain on Vertex AI RAG Templates

[Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)
(LangChain on Vertex AI) is a managed service that helps you to build and deploy
an agent reasoning framework.

## Templates

Use the following templates to deploy Retrieval Augmented Generation (RAG) applications with an AlloyDB database.

Description | Sample
----------- | ------
Deploy a pre-built `LangchainAgent` with custom RAG tool | [prebuilt_lanchain_agent_template.py](prebuilt_lanchain_agent_template.py)
Build and deploy a question-answering RAG application | [retriever_chain_template.py](retriever_chai_template.py)
Build and deploy an Agent with RAG tool and Memory | [retriever_agent_with_history_template.py](retriever_agent_with_history_template.py)

## Before you begin

1. In the Google Cloud console, on the project selector page, select or [create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. [Make sure that billing is enabled for your Google Cloud project](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console).
1. [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets).
1. Enable [AI Platform, AlloyDB, and Service Networking APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,alloydb.googleapis.com,servicenetworking.googleapis.com&_ga=2.92928541.1293093187.1719511698-1945987529.1719351858)

1. [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
1. [Configure Public IP.](https://cloud.google.com/alloydb/docs/connect-public-ip)
1. [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
1. Create a [vector store table](https://github.com/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/vector_store.ipynb) and [chat message history table](https://github.com/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/chat_message_history.ipynb).
1. Grant IAM permissions, `roles/alloydb.client`, `roles/aiplatform.user`, and `serviceusage.serviceUsageConsumer` to the AI Platform Reasoning Engine Service Agent service account: `service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com` to connect to the AlloyDB instance.
1. Use `create_embeddings.py` to add data to your vector store.
1. Open the template and add your project's values:
    ```
    PROJECT_ID = os.getenv("PROJECT_ID") or "my-project-id"
    STAGING_BUCKET = os.getenv("STAGING_BUCKET") or "gs://my-bucket"
    REGION = os.getenv("REGION") or "us-central1"
    CLUSTER = os.getenv("CLUSTER") or "my-alloy-db"
    INSTANCE = os.getenv("INSTANCE") or "my-primary"
    DATABASE = os.getenv("DATABASE") or "my_database"
    TABLE_NAME = os.getenv("TABLE_NAME") or "my_test_table"
    CHAT_TABLE_NAME = os.getenv("CHAT_TABLE_NAME") or "my_chat_table"
    USER = os.getenv("DB_USER") or "postgres"
    PASSWORD = os.getenv("DB_PASSWORD") or "password"
    ```

Learn more at [Deploying a RAG Application with AlloyDB with Reasoning Engine on Vertex AI](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/tutorial_alloydb_rag_agent.ipynb).


