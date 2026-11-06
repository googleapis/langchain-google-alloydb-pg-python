Document Compressor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBDocumentCompressor`` is a LangChain document compressor that reranks documents in the database with AlloyDB AI's ``google_ml.rank`` function. ``model_id`` is required.

.. note::
   ``google_ml.rank`` needs the ``google_ml_integration.enable_ai_query_engine`` database flag turned on, and the AlloyDB service agent needs the ``roles/discoveryengine.viewer`` IAM role to call the Vertex AI ranking API.

.. code-block:: python

    from langchain_google_alloydb_pg import AlloyDBDocumentCompressor

    compressor = AlloyDBDocumentCompressor(
        engine=engine, model_id="semantic-ranker-default-003", top_n=3
    )
    docs = compressor.compress_documents(documents, "What is AlloyDB?")

.. automodule:: langchain_google_alloydb_pg.document_compressor
  :members:
  :private-members:
  :noindex:
