# Migrate a Vector Store to AlloyDB

This guide provides step-by-step instructions on migrating data from existing vector stores to AlloyDB.

Supported Vector Stores

- Pinecone
- Weaviate
- ChromaDB
- Qdrant
- Milvus

## Prerequisites

There needs to be an AlloyDB database set up for the migration process.

How to set up AlloyDB:

- [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
- [Enable the AlloyDB API](https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com)
- [Create an AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
- [Create an AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
- [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)

Install required libraries

```bash
pip install --upgrade --quiet langchain-google-alloydb-pg langchain-core
```

> **_NOTE:_**  The langchain-core library is installed to use the Fake embeddings service. To use a different embedding service, you'll need to install the appropriate library for your chosen provider. Choose embeddings services from [LangChain's Embedding models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/).

## How to Migrate

### Step 1. **Retrieve Data from Existing Vector Database**

The process of getting data from vector stores varies depending on the specific database. Below are code snippets illustrating the process for some common stores:

#### Pinecone

   1. Install any prerequisites using the [docs](https://docs.pinecone.io/reference/python-sdk).

   2. Get pinecone index

        ```python
        from pinecone import Pinecone

        # Replace PINECONE_API_KEY
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index('index_name')
        ```

   3. Get all data from index

        ```python
        def get_all_ids():
            results = index.list_paginated(prefix="")
            ids = [v.id for v in results.vectors]
            while results.pagination is not None:
                pagination_token = results.pagination.next
                results = index.list_paginated(prefix="", pagination_token=pagination_token)
                ids.extend([v.id for v in results.vectors])
            return ids

        def get_all_data():
            all_data = index.fetch(ids=get_all_ids(index))
            ids = []
            embeddings = []
            content = []
            metadatas = []
            for doc in all_data["vectors"].values():
                ids.append(doc["id"])
                embeddings.append(doc["values"])
                content.append(doc["metadata"]["text"])
                metadata = doc["metadata"]
                del metadata["text"]
                metadatas.append(metadata)
            return ids, content, embeddings, metadatas
        ```

#### Weaviate

   1. Install any prerequisites using the [docs](https://weaviate.io/developers/weaviate/client-libraries/python#installation).

        Learn more about how you can customize the [Weaviate Cloud client](https://weaviate.io/developers/weaviate/connections/connect-cloud) and select a [Model Provider](https://weaviate.io/developers/weaviate/model-providers#api-based).

   2. Create client

        ```python
        import weaviate

        client = weaviate.connect_to_weaviate_cloud(
                cluster_url='db_url',
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
                headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
            )
        ```

   3. Get all data from collection

        ```python
        def get_all_data():
            try:
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=URL,
                    auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
                    headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
                )
                ids = []
                content = []
                embeddings = []
                metadatas = []
                collection = client.collections.get('collection_name')
                for item in collection.iterator(include_vector=True):
                    ids.append(str(item.uuid))
                    content.append(item.properties["page_content"])
                    embeddings.append(item.vector["default"])
                    metadatas.append(item.properties["metadata"])
            finally:
                client.close()
            return ids, content, embeddings, metadatas
        ```

        > **_NOTE:_**  Remember to always close the Weaviate client after use.

#### ChromaDB

   1. Install any prerequisites using the [docs](https://pypi.org/project/langchain-chroma/).

   2. Define the embeddings service that was used to create the VectorStore.

        Eg. To use the Langchain fake embeddings, use

        ```python
        from langchain_core.embeddings import FakeEmbeddings

        embeddings_service = FakeEmbeddings(size=768)
        ```

        In case you're using a different embeddings service, choose one from [LangChain's Embedding models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/).

   3. Create client

        ```python
        from langchain_chroma import Chroma

        # In case the ChromaDB data is saved locally, point the persist directory to the data path to load the existing vector store.
        vector_store = Chroma(
            collection_name='collection_name',
            embedding_function=embeddings_service,
            persist_directory="./chroma_langchain_db",
        )
        ```

   4. Get all data from collection

        ```python
        def get_all_data():
            docs = vector_store.get(include=["metadatas", "documents", "embeddings"])
            return docs["ids"], docs["documents"], docs["embeddings"], docs["metadatas"]
        ```

#### Qdrant

   1. Install any pre-requisites using the [docs](https://python-client.qdrant.tech/).

   2. Create client

        ```python
        from qdrant_client import QdrantClient

        client = QdrantClient(path="qdrant_db_path")
        ```

   3. Get all data from collection

        ```python
        def get_all_data():
            docs = client.scroll(collection_name='collection_name', with_vectors=True)
            ids = []
            content = []
            vector = []
            metadatas = []
            for doc in docs[0]:
                ids.append(doc.id)
                content.append(doc.payload["page_content"])
                vector.append(doc.vector)
                metadatas.append(doc.payload["metadata"])
            return ids, content, vector, metadatas
         ```

#### Milvus

   1. Install any prerequisites using the [docs](https://milvus.io/docs/install-pymilvus.md).

   2. Create client

        ```python
        from pymilvus import MilvusClient

        client = MilvusClient(uri='connection_uri')
        ```

   3. Get all data from collection

        ```python
        def get_all_data():
            all_docs = client.query(
                collection_name='collection_name',
                filter='pk >= "0"',
                output_fields=["pk", "col1", "col2", "text", "vector"],
            )
            ids = []
            content = []
            embeddings = []
            metadatas = []
            for doc in all_docs:
                ids.append(doc["pk"])
                content.append(doc["text"])
                embeddings.append(doc["vector"])
                del doc["pk"]
                del doc["text"]
                del doc["vector"]
                metadatas.append(doc)
            return ids, content, embeddings, metadatas
        ```

### Step 2. **Copy the data to AlloyDB**

1. Define embeddings service.

    In case you're using a different embeddings service, choose one from [LangChain's Embedding models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/).

    ```python
    from langchain_core.embeddings import FakeEmbeddings

    embeddings_service = FakeEmbeddings(size=768)
    ```

    > **_NOTE:_**  The embeddings service defined here is not used to generate the embeddings, but required by the vectorstore.
    > Embeddings are directly copied from the original table.

2. Create AlloyDB table and Vector Store

    ```python
    from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

    # Replace these variable values
    engine = await AlloyDBEngine.afrom_instance(
        project_id="my-project-id",
        instance="my-instance-name",
        region="us-central1",
        cluster="my-primary",
        database="test_db",
        user="user",
        password="password",
    )

    # Create an AlloyDB table. Set the table name.
    await engine.ainit_vectorstore_table(
        table_name='table_name',

        # Fake embeddings use a vector size of 768.
        # If you're choosing another vector embeddings service, choose the corresponding vector size
        vector_size=768,
    )

    # Create a vector store instance
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        embedding_service=embeddings_service,
        table_name='table_name',
    )
    ```

    > **_NOTE:_** This code adds metadata to the "langchain_metadata" column in a JSON format. For more efficient filtering, you can organize this metadata into separate columns. Refer to the [vector store docs](https://github.com/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/vector_store.ipynb) for examples of creating metadata columns.

    > **_NOTE:_** The weaviate examples here use Cohere Embeddings, which have a size of 1024. Make sure to change the vector size while creating the table.

3. Insert data to AlloyDB

    ```python
    ids, content, embeddings, metadatas = get_all_data()
    await vector_store.aadd_embeddings(
        texts=content,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    ```

### Step 3. **Delete data from existing Vector Database**

1. Verify data copy

    ```python
    from langchain_google_alloydb_pg import AlloyDBLoader

    loader = await AlloyDBLoader.create(
        engine=engine,
        query=f"SELECT * FROM table_name;",
        content_columns="content",
        metadata_json_column="langchain_metadata"
    )

    documents = loader.load()
    assert len(documents) == len(ids)
    ```

2. Delete existing data in the collection

   #### Pinecone

    [Source doc](https://docs.pinecone.io/guides/indexes/delete-an-index)

    ```python
    pc.delete_index('index_name')
    ```

   #### ChromaDB

    [Source doc](https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete_collection)

    ```python
    vector_store.delete_collection('collection_name')
    ```

   #### Qdrant

    [Source doc](https://python-client.qdrant.tech/qdrant_client.qdrant_client)

    ```python
    client.delete_collection('collection_name')
    ```

   #### Milvus

    [Source doc](https://milvus.io/docs/v2.0.x/drop_collection.md)

    ```python
    from pymilvus import utility

    utility.drop_collection('collection_name')
    ```

   #### Weaviate

    [Source doc](https://weaviate.io/developers/weaviate/manage-data/collections#delete-a-collection)

    ```python
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=URL,
            auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
            headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
        )
        client.collections.delete('collection_name')
    finally:
        client.close()
    ```
