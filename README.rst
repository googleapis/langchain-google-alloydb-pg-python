AlloyDB for PostgreSQL for LangChain
==================================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

The **AlloyDB for PostgreSQL for LangChain** package provides a first class experience for connecting to
AlloyDB instances from the LangChain ecosystem while providing the following benefits:

- **Simplified & Secure Connections**: easily and securely create shared connection pools to connect to Google Cloud databases utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Improved performance & Simplified management**: a single-table schema leads to faster query execution, especially for large collections.
- **Improved metadata handling**: store metadata in columns instead of JSON, resulting in significant performance improvements.
- **Clear separation**: clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.
- **Better integration with AlloyDB**: built-in methods to take advantage of AlloyDB's advanced indexing, in-database AI functions, and scalability capabilities.

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#stability-levels
.. |pypi| image:: https://img.shields.io/pypi/v/langchain-google-alloydb-pg.svg
   :target: https://pypi.org/project/langchain-google-alloydb-pg/
.. |versions| image:: https://img.shields.io/pypi/pyversions/langchain-google-alloydb-pg.svg
   :target: https://pypi.org/project/langchain-google-alloydb-pg/
.. _Client Library Documentation: https://cloud.google.com/python/docs/reference/langchain-google-alloydb-pg/latest
.. _Product Documentation: https://cloud.google.com/alloydb


Quick Start
-----------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the AlloyDB API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the AlloyDB API.: https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip. `virtualenv`_ is a tool to create isolated Python environments. The basic problem it addresses is
one of dependencies and versions, and indirectly permissions.

With `virtualenv`_, it's
possible to install this library without needing system install
permissions, and without clashing with the installed system
dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.10

Mac/Linux
^^^^^^^^^

.. code-block:: console

   pip install virtualenv
   virtualenv <your-env>
   source <your-env>/bin/activate
   <your-env>/bin/pip install langchain-google-alloydb-pg

Windows
^^^^^^^

.. code-block:: console

    pip install virtualenv
    virtualenv <your-env>
    <your-env>\Scripts\activate
    <your-env>\Scripts\pip.exe install langchain-google-alloydb-pg



Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a vector store to store embedded data and perform vector search.

.. code-block:: python

   from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
   from langchain_google_vertexai import VertexAIEmbeddings


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   embeddings_service = VertexAIEmbeddings(model_name="text-embedding-005")
   vectorstore = AlloyDBVectorStore.create_sync(
       engine,
       table_name="my-table",
       embedding_service=embeddings_service
   )

Hybrid Search
~~~~~~~~~~~~~

The ``AlloyDBVectorStore`` supports hybrid search (dense vectors + full text) for more comprehensive and relevant search results.

.. code-block:: python

  from langchain_google_alloydb_pg import HybridSearchConfig, reciprocal_rank_fusion

  # Continues from the Vector Store example above.
  vs = AlloyDBVectorStore.create_sync(
      engine=engine,
      table_name="my-table",
      embedding_service=embeddings_service,
      hybrid_search_config=HybridSearchConfig(
        fusion_function=reciprocal_rank_fusion
      ),
  )
  hybrid_docs = vs.similarity_search("products", k=5)

Vector Indexing with ScaNN
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ScaNN index makes similarity search fast by sorting the table's vectors into groups
(called *leaves*) when the index is built, and then only looking inside the most promising
groups at query time instead of comparing against every row. That leaves you with two
decisions: how many groups to build, and how many groups to look inside per query.

**1. Build time: how many groups?** Use ``ScaNNIndex``.

.. list-table::
   :header-rows: 1

   * - You write
     - What happens
   * - ``ScaNNIndex(num_leaves=N)``
     - You pick the number of groups. AlloyDB recommends ``sqrt(rows)`` for a balance of
       build time and quality, or ``rows / 100`` for best quality.
   * - ``ScaNNIndex(mode="AUTO")``
     - AlloyDB picks the number of groups from the table size and tunes it over time.
       ``num_leaves`` is ignored. On tables with fewer than 10,000 rows the build is
       deferred until the table reaches 10,000 rows.
   * - ``ScaNNIndex(num_leaves=N, mode="MANUAL")``
     - Same as the first form, written explicitly.

Index creation needs enough ``maintenance_work_mem``; ``set_maintenance_work_mem`` computes
and sets it for you.

.. code-block:: python

   from langchain_google_alloydb_pg.indexes import ScaNNIndex, ScaNNQueryOptions

   # Continues from the Vector Store example above (768-dimensional vectors).

   # Option A: you choose the number of groups (here, sqrt of ~10,000 rows).
   index = ScaNNIndex(num_leaves=100)
   vectorstore.set_maintenance_work_mem(index.num_leaves, vector_size=768)
   vectorstore.apply_vector_index(index)

   # Option B: let AlloyDB choose and tune the number of groups.
   vectorstore.apply_vector_index(ScaNNIndex(name="my_table_auto_idx", mode="AUTO"))

**2. Query time: how many groups to look inside?** Use ``ScaNNQueryOptions``. Looking inside
more groups finds more of the true nearest neighbors (higher recall) but takes longer.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``num_leaves_to_search=N``
     - Look inside ``N`` groups (default ``1``). Raise it until recall meets your target;
       ``0`` lets the server choose.
   * - ``pct_leaves_to_search=P``
     - Look inside ``P`` percent of the groups. Use this with ``mode="AUTO"`` indexes, where
       you don't know how many groups exist.
   * - ``pre_reordering_num_neighbors=M``
     - Advanced: how many candidates to collect before exact re-scoring. Leave the default
       (``-1``) unless you are tuning recall on high-dimensional vectors.

.. code-block:: python

   # Manually tuned index: look inside 10 of the 100 groups.
   vectorstore = AlloyDBVectorStore.create_sync(
       engine,
       table_name="my-table",
       embedding_service=embeddings_service,
       index_query_options=ScaNNQueryOptions(num_leaves_to_search=10),
   )

   # Auto-tuned index: look inside 10% of the groups, however many there are.
   auto_vectorstore = AlloyDBVectorStore.create_sync(
       engine,
       table_name="my-table",
       embedding_service=embeddings_service,
       index_query_options=ScaNNQueryOptions(pct_leaves_to_search=10),
   )

For the full tuning procedure, see `Tune ScaNN indexes`_ and `Create a ScaNN index`_.

.. _`Tune ScaNN indexes`: https://cloud.google.com/alloydb/docs/ai/best-practices-tuning-scann
.. _`Create a ScaNN index`: https://cloud.google.com/alloydb/docs/ai/create-scann-index

See the full `Vector Store`_ tutorial.

.. _`Vector Store`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/vector_store.ipynb


Document Loader Usage
~~~~~~~~~~~~~~~~~~~~~

Use a document loader to load data as LangChain ``Document``\ s.

.. code-block:: python

   from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBLoader


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   loader = AlloyDBLoader.create_sync(
       engine,
       table_name="my-table-name"
   )
   docs = loader.lazy_load()

See the full `Document Loader`_ tutorial.

.. _`Document Loader`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/document_loader.ipynb

Chat Message History Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ChatMessageHistory`` to store messages and provide conversation
history to LLMs.

.. code:: python

   from langchain_google_alloydb_pg import AlloyDBChatMessageHistory, AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   history = AlloyDBChatMessageHistory.create_sync(
       engine,
       table_name="my-message-store",
       session_id="my-session-id"
   )

See the full `Chat Message History`_ tutorial.

.. _`Chat Message History`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/chat_message_history.ipynb

LangGraph Checkpoint Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``AlloyDBSaver`` to save snapshots of the graph state at a given point in time.

.. code:: python

   from langchain_google_alloydb_pg import AlloyDBSaver, AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   checkpoint = AlloyDBSaver.create_sync(engine)

See the full `Checkpoint`_ tutorial.

.. _`Checkpoint`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/langgraph_checkpoint.ipynb

AlloyDB AI Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

The embeddings, document reranking, generative AI tools, forecasting, and auto-embedding
features below call AlloyDB AI functions inside the database. Before you use them:

- `Integrate your AlloyDB cluster with Vertex AI`_ and create the ``google_ml_integration``
  extension in your database (``CREATE EXTENSION IF NOT EXISTS google_ml_integration``).
- Use a model that is `pre-registered in AlloyDB`_ (for example ``text-embedding-005``), or
  register your own with ``AlloyDBModelManager``. ``AlloyDBEmbeddings`` and
  ``AlloyDBModelManager`` both need the ``google_ml_integration.enable_model_support`` database
  flag turned on.
- Some functions need additional database flags, a minimum extension or PostgreSQL version, or
  IAM roles. For example, ``google_ml.rank`` needs ``google_ml_integration.enable_ai_query_engine``;
  ``google_ml.analyze_sentiment`` and ``google_ml.summarize`` need PostgreSQL 17 or later and
  ``google_ml_integration.enable_preview_ai_functions``; ``AlloyDBSummaryTool`` needs
  ``google_ml_integration`` 1.6.1 or later. See the `AlloyDB AI documentation`_ and the API
  Reference pages for each component.

.. _`Integrate your AlloyDB cluster with Vertex AI`: https://cloud.google.com/alloydb/docs/ai/configure-vertex-ai
.. _`pre-registered in AlloyDB`: https://cloud.google.com/alloydb/docs/reference/supported-ai-models
.. _`AlloyDB AI documentation`: https://cloud.google.com/alloydb/docs/ai

Embeddings & Model Endpoint Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``AlloyDBEmbeddings`` to generate text embeddings and multimodal image embeddings directly
inside AlloyDB with ``google_ml_integration``. Unlike ``VertexAIEmbeddings``, the database calls
the model, so you can use the same model from SQL and from LangChain.

.. code-block:: python

   from langchain_google_alloydb_pg import AlloyDBEmbeddings, AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")

   embeddings = AlloyDBEmbeddings.create_sync(engine=engine, model_id="text-embedding-005")
   query_vector = embeddings.embed_query("What is AlloyDB?")

   # Image embeddings need a registered multimodal model, such as multimodalembedding@001.
   image_embeddings = AlloyDBEmbeddings.create_sync(engine=engine, model_id="multimodalembedding@001")
   image_vector = image_embeddings.embed_image("gs://my-bucket/image.jpg")

Use ``AlloyDBModelManager`` (an async API) to list, register, and drop model endpoints. See the
full `Model Endpoint Management`_ tutorial.

.. _`Model Endpoint Management`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/model_endpoint_management.ipynb

Document Reranking Usage
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``AlloyDBDocumentCompressor`` to rerank retrieved documents in the database with
``google_ml.rank``. Reranking adds a model call per query, so it is most useful when retrieval
returns many similar candidates and answer quality matters more than latency.

.. code-block:: python

   from langchain_core.documents import Document
   from langchain_google_alloydb_pg import AlloyDBDocumentCompressor, AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   compressor = AlloyDBDocumentCompressor(
       engine=engine,
       model_id="semantic-ranker-default-003",
       top_n=3,
   )
   docs = [
       Document(page_content="AlloyDB is a PostgreSQL-compatible database."),
       Document(page_content="Cloud Storage stores objects."),
   ]
   reranked_docs = compressor.compress_documents(docs, "What is AlloyDB?")

To rerank the results of a retriever automatically, wrap it in LangChain's
``ContextualCompressionRetriever`` with ``base_compressor=compressor``.

Generative AI Tools Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``AlloyDBSentimentTool``, ``AlloyDBSummaryTool``, and ``AlloyDBIfTool`` to call AlloyDB AI
functions (``google_ml.analyze_sentiment``, ``google_ml.summarize``, and ``google_ml.if``) from
LangChain agents or chains. ``model_id`` is fixed when you create the tool, so the agent can't
change it; omit it to use the database's default model.

.. code-block:: python

   from langchain_google_alloydb_pg import (
       AlloyDBEngine,
       AlloyDBIfTool,
       AlloyDBSentimentTool,
       AlloyDBSummaryTool,
   )


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")

   sentiment_tool = AlloyDBSentimentTool(engine=engine, model_id="gemini-2.5-flash")
   sentiment = sentiment_tool.invoke({"content": "I love this database!"})  # "positive"

   summary_tool = AlloyDBSummaryTool(engine=engine, model_id="gemini-2.5-flash")
   summary = summary_tool.invoke(
       {"content": "Long article text...", "additional_instructions": "In one sentence"}
   )

   if_tool = AlloyDBIfTool(engine=engine, model_id="gemini-2.5-flash")
   is_match = if_tool.invoke({"condition": "Is AlloyDB compatible with PostgreSQL?"})  # True

Engine AI & Performance Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBEngine`` also exposes in-database AI and performance operations. Each method has an
async variant prefixed with ``a`` (for example ``aforecast``).

.. code-block:: python

   from langchain_google_alloydb_pg import AlloyDBEngine


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")

   # Generate embeddings for every row of an existing table (ai.initialize_embeddings).
   # Needs google_ml_integration 1.5.6+ with the enable_model_support and
   # enable_faster_embedding_generation flags on. The embedding column must already
   # exist as a vector(N) column; pass overwrite=True to replace existing values.
   engine.initialize_embeddings(
       table_name="products",
       model_id="text-embedding-005",
       content_column="description",
       embedding_column="embedding",
   )

   # Time-series forecasting (google_ml.forecast). The model must be deployed and
   # registered with google_ml.create_model(model_type => 'ts_forecasting'); see
   # https://cloud.google.com/alloydb/docs/ai/perform-time-series-forecasting
   forecast_rows = engine.forecast(
       "timesfm_v2",
       table_name="metrics",
       timestamp_column="ts",
       data_column="requests",
       horizon=24,
       conf_level=0.95,
   )

   # Vector Assist: get and apply index tuning recommendations for an embedding column.
   # Needs `CREATE EXTENSION vector_assist CASCADE;` in the database. Pass the model
   # that generated the embeddings when the column is already populated.
   engine.define_vector_assist_spec(
       table_name="products",
       embedding_column="embedding",
       embedding_model="text-embedding-005",
   )
   recommendations = engine.get_vector_assist_recommendations(
       table_name="products", embedding_column="embedding"
   )
   engine.apply_vector_assist_spec(table_name="products", embedding_column="embedding")

   # Columnar engine: add columns to the columnar store, or let AlloyDB recommend them.
   # Needs the columnar engine enabled on the instance (google_columnar_engine.enabled).
   engine.enable_columnar_engine(table_name="products", columns=["description"])
   engine.run_auto_columnarization()

Example Usage
-------------

Code examples can be found in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/samples

Converting between Sync & Async Usage
-------------------------------------

Async functionality improves the speed and efficiency of database connections through concurrency,
which is key for providing enterprise quality performance and scaling in GenAI applications. This
package uses a native async Postgres driver, `asyncpg`_, to optimize Python's async functionality.

LangChain supports `async programming`_, since LLM based application utilize many I/O-bound operations,
such as making API calls to language models, databases, or other services. All components should provide
both async and sync versions of all methods.

`asyncio`_ is a Python library used for concurrent programming and is used as the foundation for multiple
Python asynchronous frameworks. asyncio uses `async` / `await` syntax to achieve concurrency for
non-blocking I/O-bound tasks using one thread with cooperative multitasking instead of multi-threading.

.. _`async programming`: https://python.langchain.com/docs/concepts/async/
.. _`asyncio`: https://docs.python.org/3/library/asyncio.html
.. _`asyncpg`: https://github.com/MagicStack/asyncpg

Converting Sync to Async
~~~~~~~~~~~~~~~~~~~~~~~~

Update sync methods to `await` async methods

.. code:: python

   engine = await AlloyDBEngine.afrom_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   await engine.ainit_vectorstore_table(table_name="my-table", vector_size=768)
   vectorstore = await AlloyDBVectorStore.create(
      engine,
      table_name="my-table",
      embedding_service=VertexAIEmbeddings(model_name="text-embedding-005")
   )

Run the code: notebooks
^^^^^^^^^^^^^^^^^^^^^^^

ipython and jupyter notebooks support the use of the `await` keyword without any additional setup

Run the code: FastAPI
^^^^^^^^^^^^^^^^^^^^^

Update routes to use `async def`.

.. code:: python

   @app.get("/invoke/")
   async def invoke(query: str):
      return await retriever.ainvoke(query)


Run the code: Local python file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to create a top-level async method definition: `async def` to wrap multiple async methods.
Then use `asyncio.run()` to run the top-level entrypoint, e.g. "main()"

.. code:: python

   async def main():
      response = await retriever.ainvoke(query)
      print(response)

   asyncio.run(main())


Contributions
-------------

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ and `DEVELOPER`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CONTRIBUTING.md
.. _`DEVELOPER`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/DEVELOPER.md
.. _`Code of Conduct`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
