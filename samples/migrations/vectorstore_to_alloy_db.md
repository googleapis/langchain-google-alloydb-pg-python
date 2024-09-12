# Vector Store to AlloyDB migration

This guide provides step-by-step instructions on migrating data from existing vector stores to the AlloyDB vectorstore.

Supported Vector Stores

- Pinecone
- Weaviate
- ChromaDB
- Qdrant
- Milvus

## How to Migrate

1. **Retrieve Data from Existing Vector Database**

    The process of getting data from vector stores varies depending on the specific database. Below are code snippets illustrating the process for some common stores:

   ### Pinecone

    Get pinecone index

    ```python
    from pinecone import Pinecone

    # Replace PINECONE_API_KEY
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index('index_name')
    ```

    Get all data from index

    ```python
    def get_all_ids(index):
        results = index.list_paginated(prefix="")
        ids = [v.id for v in results.vectors]
        while results.pagination is not None:
            pagination_token = results.pagination.next
            results = index.list_paginated(prefix="", pagination_token=pagination_token)
            ids.extend([v.id for v in results.vectors])
        return ids

    def get_all_data(index):
        all_data = index.fetch(ids=get_all_pinecone_ids(index))
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

   ### ChromaDB

   ### Qdrant

   ### Milvus

2. **Copy the data to AlloyDB**
    1. Set up an AlloyDB database
        - [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
        - [Enable the AlloyDB API](https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com)
        - [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
        - [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
        - [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)
    2. Transfer data

        1. Install libraries

            ```bash
            pip install --upgrade --quiet  langchain-google-alloydb-pg langchain-google-vertexai
            ```

        2. Define embeddings service.

            In case you're using a different embeddings service, choose one from <https://python.langchain.com/v0.2/docs/integrations/text_embedding/>

            ```python
            from langchain_google_vertexai import VertexAIEmbeddings

            embeddings_service = VertexAIEmbeddings(
                model_name="textembedding-gecko@003", project=PROJECT_ID
            )
            ```

        3. Create AlloyDB table and Vector Store

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

        4. Insert data to AlloyDB

            ```python
            ids, embeddings, content, metadatas = get_all_data('collection_name')
            await vector_store.aadd_embeddings(
                texts=content,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            ```

3. Delete data from existing Vector Database
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

       ## Pinecone

       ```python
        index.delete(ids=ids)
       ```

       ## ChromaDB

       ## Qdrant

       ## Milvus

       ## Weaviate
