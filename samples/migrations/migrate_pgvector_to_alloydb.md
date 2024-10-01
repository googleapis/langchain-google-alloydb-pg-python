# Migrate a `PGVector` vector store to `AlloyDBVectorStore`

This guide shows how to migrate from the [`PGVector`](https://github.com/langchain-ai/langchain-postgres) vector store class to the [`AlloyDBVectorStore`](https://github.com/googleapis/langchain-google-alloydb-pg-python) class.

## Why migrate?

The PGVector interface uses a two-table schema to store vector data and collection metadata.  This approach can be less efficient and harder to manage compared to the single-table schema used by the AlloyDB interface.

Migrating to the AlloyDB interface provides the following benefits:

- **Simplified management**: A single table contains data corresponding to a single collection, making it easier to query, update, and maintain.
- **Improved metadata handling**: AlloyDB stores metadata in columns instead of JSON, resulting in significant performance improvements.
- **Schema flexibility**: The interface allows users to add tables into any database schema.
- **Improved performance**: The single-table schema can lead to faster query execution, especially for large collections.
- **Enhanced security**: Easily and securely connect to AlloyDB utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Better integration with AlloyDB**: Take advantage of AlloyDB's advanced indexing and scalability capabilities.
- **Clear separation**: AlloyDB classes clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.

## Before you begin

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

## How to migrate

1. Create an AlloyDB engine.

    ```python
    from langchain_google_alloydb_pg import AlloyDBEngine

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
    ```

    > **_NOTE:_** All async methods have corresponding sync methods.

2. Create a new table to migrate existing data.

    ```python
    # Vertex AI embeddings uses a vector size of 768. Change this according to your embeddings service.
    VECTOR_SIZE = 768

    await engine.ainit_vectorstore_table(
        table_name="destination_table",
        vector_size=VECTOR_SIZE,
    )
    ```

    When creating your vectorstore table, you have the flexibility to define custom metadata and ID columns. This is particularly useful for:

    - **Filtering**: Metadata columns allow you to easily filter your data within the vectorstore. For example, you might store the document source, date, or author as metadata for efficient retrieval.
    - **Non-UUID Identifiers**: By default, the id_column uses UUIDs. If you need to use a different type of ID (e.g., an integer or string), you can define a custom id_column.

    Here's how to customize your table:

    ```python
    metadata_columns = [
        Column(f"col_0_{collection_name}", "VARCHAR"),
        Column(f"col_1_{collection_name}", "VARCHAR"),
    ]
    await engine.ainit_vectorstore_table(
        table_name="destination_table",
        vector_size=VECTOR_SIZE,
        metadata_columns=metadata_columns,
        id_column=Column("langchain_id", "VARCHAR"),
    )
    ```

    You can refer to the [API Reference](https://cloud.google.com/python/docs/reference/langchain-google-alloydb-pg/latest/langchain_google_alloydb_pg.engine.AlloyDBEngine#langchain_google_alloydb_pg_engine_AlloyDBEngine_ainit_vectorstore_table) for any vector store customisations.

3. Create a vector store object to interact with the new data.

    > **_NOTE:_** The Fake Embeddings embedding service is only used to initialise a vector store object, not to generate any embeddings. The embeddings are directly copied from the PGVector database.

    ```python
    from langchain_google_alloydb_pg import AlloyDBVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    vector_store = await AlloyDBVectorStore.create(
        engine,
        embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        table_name="destination_table",
    )
    ```

    If you have any customisations on the metadata or the id columns, add them to the vector store as follows:

    ```python
    from langchain_google_alloydb_pg import AlloyDBVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    vector_store = await AlloyDBVectorStore.create(
        engine,
        embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        table_name="destination_table",
        metadata_columns=[col.name for col in metadata_columns],
        id_column="langchain_id",
    )
    ```

4. Migrate data to the new table.

    ```python
    from langchain_google_alloydb_pg.utils.pgvector_migrator import amigrate_pgvector_collection

    await amigrate_pgvector_collection(
        engine,
        # Set collection name here
        collection_name="collection_name",
        vector_store=vector_store,
        # This deletes data from the original table upon migration. You can choose to turn it off.
        delete_pg_collection=True,
    )
    ```

    The data will only be deleted from the original table once all of it has been successfully copied to the destination table.

> **TIP:** If you would like to migrate multiple collections, you can use the `alist_pgvector_collection_names` method to get the names of all collections, allowing you to iterate through them.
>
> ```python
> from langchain_google_alloydb_pg.utils.pgvector_migrator import alist_pgvector_collection_names
> 
> all_collection_names = await alist_pgvector_collection_names(engine)
> print(all_collection_names)
> ```
