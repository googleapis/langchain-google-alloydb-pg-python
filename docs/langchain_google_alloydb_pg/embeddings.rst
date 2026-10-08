Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBEmbeddings`` generates text and image embeddings directly inside AlloyDB using the ``google_ml_integration`` extension (``embedding()`` for text queries and ``google_ml.image_embedding()`` for image URIs or base64-encoded images).

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

.. automodule:: langchain_google_alloydb_pg.embeddings
  :members:
  :private-members:
  :noindex:
