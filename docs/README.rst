AlloyDB for PostgreSQL for LangChain
====================================

This package contains the
`LangChain <https://github.com/langchain-ai/langchain>`_ integrations
for AlloyDB for PostgreSQL.

   **🧪 Preview:** This feature is covered by the Pre-GA Offerings Terms
   of the Google Cloud Terms of Service. Please note that pre-GA
   products and features might have limited support, and changes to
   pre-GA products and features might not be compatible with other
   pre-GA versions. For more information, see the `launch stage
   descriptions <https://cloud.google.com/products#product-launch-stages>`_

-  `Documentation <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs>`_
-  `API Reference <https://cloud.google.com/python/docs/reference/langchain-google-alloydb-pg/latest>`_

Getting Started
---------------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform
   project. <https://console.cloud.google.com/project>`_
2. `Enable billing for your
   project. <https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project>`_
3. `Enable the Google Cloud AlloyDB
   API. <https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com>`_
4. `Setup
   Authentication. <https://googleapis.dev/python/google-api-core/latest/auth.html>`_

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip. `virtualenv`_ is a tool to create isolated Python environments. The basic problem it addresses is
one of dependencies and versions, and indirectly permissions.

With `virtualenv`_, it’s
possible to install this library without needing system install
permissions, and without clashing with the installed system
dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

.. code:: bash

   pip install virtualenv
   virtualenv <your-env>
   source <your-env>/bin/activate
   <your-env>/bin/pip install langchain-google-alloydb-pg

Vector Store Usage
------------------

Use a vector store to store embedded data and perform vector search.

.. code:: python

   from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
   from langchain_google_vertexai import VertexAIEmbeddings


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   embeddings_service = VertexAIEmbeddings()
   vectorstore = AlloyDBVectorStore(
       engine,
       table_name="my-table",
       embeddings=embedding_service
   )

See the full `Vector
Store <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/vector_store.ipynb>`_
tutorial.

Document Loader Usage
---------------------

Use a document loader to load data as LangChain ``Document``\ s.

.. code:: python

   from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBLoader


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   loader = PostgresSQLLoader(
       engine,
       table_name="my-table-name"
   )
   docs = loader.lazy_load()

See the full `Document
Loader <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/document_loader.ipynb>`_
tutorial.

Chat Message History Usage
--------------------------

Use ``ChatMessageHistory`` to store messages and provide conversation
history to LLMs.

.. code:: python

   from langchain_google_alloydb_pg import AlloyDBChatMessageHistory, AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   history = AlloyDBChatMessageHistory(
       engine,
       table_name="my-message-store",
       session_id="my-session-id"
   )

See the full `Chat Message
History <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/chat_message_history.ipynb>`_
tutorial.

Contributing
------------

Contributions to this library are always welcome and highly encouraged.

See
`CONTRIBUTING <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CONTRIBUTING.md>`_
for more information how to get started.

Please note that this project is released with a Contributor Code of
Conduct. By participating in this project you agree to abide by its
terms. See `Code of
Conduct <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CODE_OF_CONDUCT.md>`_
for more information.

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
