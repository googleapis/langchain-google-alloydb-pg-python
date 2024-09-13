# Vector Store to AlloyDB migration

This guide provides step-by-step instructions on migrating data from existing vector stores to the AlloyDB vectorstore.

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
- [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
- [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
- [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)

Install required libraries

```bash
pip install --upgrade --quiet  langchain-google-alloydb-pg langchain-google-vertexai
```

## How to Migrate

1. **Retrieve Data from Existing Vector Database**

    The process of getting data from vector stores varies depending on the specific database. Below are code snippets illustrating the process for some common stores:

   ### Pinecone

   Install any prerequisites using the [docs](https://docs.pinecone.io/reference/python-sdk).

    Get pinecone index

    ```python
    from pinecone import Pinecone

    # Replace PINECONE_API_KEY
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index('index_name')
    ```

    Get all data from index

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

   ### Weaviate

   Install any prerequisites using the [docs](https://weaviate.io/developers/weaviate/client-libraries/python#installation)

    Create client

   ```python
   import weaviate

   client = weaviate.connect_to_weaviate_cloud(
        cluster_url='db_url',
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-Cohere-Api-Key": EMBEDDINGS_API_KEY},
    )
   ```

   You can choose other embeddings types as well. ([ref](https://weaviate.io/developers/weaviate/model-providers#api-based))

   Get all data from collection

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
        collection = client.collections.get(COLLECTION_NAME)
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

   ### ChromaDB

   Install any prerequisites using the [docs](https://pypi.org/project/langchain-chroma/).

   Define the embeddings service that was used to create the VectorStore.

   Eg. To use VertexAI embedding service, use

   ```python
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings_service = VertexAIEmbeddings(
        model_name="textembedding-gecko@003", project=PROJECT_ID
    )
   ```

    In case you're using a different embeddings service, choose one from <https://python.langchain.com/v0.2/docs/integrations/text_embedding/>

   Create client

    ```python
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name='collection_name',
        embedding_function=embeddings_service,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
    )
   ```

   Get all data from collection

   ```python
   def get_all_data():
        docs = vector_store.get(include=["metadatas", "documents", "embeddings"])
        return docs["ids"], docs["documents"], docs["embeddings"], docs["metadatas"]
   ```

   ### Qdrant

   Install any pre-requisites using the [docs](https://python-client.qdrant.tech/).

   Create client

    ```python
    from qdrant_client import QdrantClient

    client = QdrantClient(path="qdrant_db_path")
    ```

    Get all data from collection

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

   ### Milvus

   Install any prerequisites using the [docs](https://milvus.io/docs/install-pymilvus.md).

   Create client

    ```python
    from pymilvus import MilvusClient

    client = MilvusClient(uri='connection_uri')
    ```

    Get all data from collection

    ```python
    def get_all_data():
        all_docs = client.query(
            collection_name='collection_name',
            filter='pk >= "0"',
            output_fields=["pk", "source", "location", "text", "vector"],
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

2. **Copy the data to AlloyDB**
    1. Define embeddings service.

        In case you're using a different embeddings service, choose one from <https://python.langchain.com/v0.2/docs/integrations/text_embedding/>

        ```python
        from langchain_google_vertexai import VertexAIEmbeddings

        embeddings_service = VertexAIEmbeddings(
            model_name="textembedding-gecko@003", project=PROJECT_ID
        )
        ```

        > **_NOTE:_**  The embeddings service defined here is not used to generate the embeddings, but required by the vectorstore.
        > Embeddings are directly copied from the original table.

    2. Create AlloyDB table and Vector Store

        ```python
        from langchain_google_alloydb_pg import (
            AlloyDBEngine,
            Column,
            AlloyDBVectorStore,
        )

        # Replace these variable values
        engine = await AlloyDBEngine.afrom_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE_NAME,
            region=REGION,
            cluster=CLUSTER,
            database=DATABASE,
            user=USER,
            password=PASSWORD,
        )

        # Create an AlloyDB table
        await engine.ainit_vectorstore_table(
            table_name='collection_name', 
            
            # VertexAI embeddings use a vector size of 768. If you're choosing another vector embeddings service, choose their corresponding vector size
            vector_size=768,

            # Define your metadata columns
            metadata_columns=[
                Column("source", "VARCHAR"), 
                Column("location", "VARCHAR")
            ],
        )

        # Create a vector store instance
        vector_store = await AlloyDBVectorStore.create(
            engine=engine,
            embedding_service=embeddings_service,
            table_name='collection_name',

            # Metadata column names
            metadata_columns=["source", "location"],
        )
        ```

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

3. **Delete data from existing Vector Database**
    1. Verify data copy

        ```python
        from langchain_google_alloydb_pg import AlloyDBLoader

        loader = await AlloyDBLoader.create(
            engine=engine,
            query=f"SELECT * FROM collection_name;",
            content_columns="content",
            metadata_columns=["source", "location"]
        )
        
        documents = loader.load()
        assert len(documents) == len(ids)
        ```

    2. Delete existing data in the collection

       ### Pinecone

       [Source doc](https://docs.pinecone.io/guides/indexes/delete-an-index)

       ```python
        pc.delete_index('index_name')
       ```

       ### ChromaDB

       [Source doc](https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete_collection)

       ```python
       vector_store.delete_collection('collection_name')
       ```

       ### Qdrant

       [Source doc](https://python-client.qdrant.tech/qdrant_client.qdrant_client)

       ```python
       client.delete_collection('collection_name')
       ```

       ### Milvus

       [Source doc](https://milvus.io/docs/v2.0.x/drop_collection.md)

        ```python
        from pymilvus import utility
        
        utility.drop_collection('collection_name')
        ```

       ### Weaviate

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
