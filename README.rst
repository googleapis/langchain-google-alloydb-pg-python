AlloyDB for PostgreSQL for LangChain
==================================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

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

With `virtualenv`_, it’s
possible to install this library without needing system install
permissions, and without clashing with the installed system
dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.8

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

Example Usage
-------------

Code samples and snippets live in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/samples


Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a vector store to store embedded data and perform vector search.

.. code-block:: python

   from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
   from langchain_google_vertexai import VertexAIEmbeddings


   engine = AlloyDBEngine.from_instance("project-id", "region", "my-cluster", "my-instance", "my-database")
   embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@003")
   vectorstore = AlloyDBVectorStore.create_sync(
       engine,
       table_name="my-table",
       embedding_service=embeddings_service
   )

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
--------------------------

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


Contributions
~~~~~~~~~~~~~

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
