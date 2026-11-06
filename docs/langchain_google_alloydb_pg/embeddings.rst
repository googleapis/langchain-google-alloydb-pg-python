Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBEmbeddings`` generates text and image embeddings directly inside AlloyDB using the ``google_ml_integration`` extension: ``embedding()`` for text and ``google_ml.image_embedding()`` for image URIs or base64-encoded images. Because the database calls the model, you can use the same model from SQL (for example, with ``AlloyDBEngine.initialize_embeddings``) and from LangChain.

.. note::
   Requires the ``google_ml_integration`` extension (1.5.3 or later), `Vertex AI integration <https://cloud.google.com/alloydb/docs/ai/configure-vertex-ai>`__ on the cluster, and the ``google_ml_integration.enable_model_support`` database flag turned on (``create`` / ``create_sync`` check for the model through ``AlloyDBModelManager``). ``model_id`` must be a `pre-registered model <https://cloud.google.com/alloydb/docs/reference/supported-ai-models>`__ such as ``text-embedding-005``, or a model registered with ``AlloyDBModelManager``. Image embeddings need a registered multimodal model such as ``multimodalembedding@001``.

.. code-block:: python

    from langchain_google_alloydb_pg import AlloyDBEmbeddings

    embeddings = AlloyDBEmbeddings.create_sync(
        engine=engine, model_id="text-embedding-005"
    )
    text_vector = embeddings.embed_query("What is AlloyDB?")

    image_embeddings = AlloyDBEmbeddings.create_sync(
        engine=engine, model_id="multimodalembedding@001"
    )
    image_vector = image_embeddings.embed_image("gs://my-bucket/image.jpg")

``embed_documents`` is not implemented and raises ``NotImplementedError``. ``AlloyDBVectorStore.add_texts`` and ``add_documents`` still work with ``AlloyDBEmbeddings``: the vector store uses ``embed_query_inline_template`` to generate the embeddings inside the ``INSERT`` statement. To embed an existing table, use ``AlloyDBEngine.initialize_embeddings``.

.. automodule:: langchain_google_alloydb_pg.embeddings
  :members:
  :private-members:
  :noindex:
