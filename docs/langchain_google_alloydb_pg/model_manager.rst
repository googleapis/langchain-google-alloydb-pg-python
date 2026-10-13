Model Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBModelManager`` lists, registers, retrieves, and drops model endpoints in AlloyDB AI `model endpoint management <https://cloud.google.com/alloydb/docs/ai/model-endpoint-overview>`_. Registered models can then be used by ``AlloyDBEmbeddings``, by the AI tools, and directly from SQL.

.. note::
   Requires the ``google_ml_integration.enable_model_support`` database flag turned on. ``create`` / ``create_sync`` install the ``google_ml_integration`` extension if it is missing, and raise if the installed version is older than 1.5.3 or the flag is off. Google `pre-registered models <https://cloud.google.com/alloydb/docs/reference/supported-ai-models>`_ don't need to be registered.

The management methods are async; create the manager with ``create`` or ``create_sync``.

.. code-block:: python

    from langchain_google_alloydb_pg import AlloyDBModelManager

    manager = await AlloyDBModelManager.create(engine)

    models = await manager.alist_models()
    model = await manager.aget_model(model_id="text-embedding-005")

    await manager.acreate_model(
        model_id="my-embedding-model",
        model_provider="google",
        model_type="text_embedding",
        model_qualified_name="text-embedding-005",
    )
    await manager.adrop_model(model_id="my-embedding-model")

See the full `Model Endpoint Management <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/model_endpoint_management.ipynb>`_ tutorial.

.. automodule:: langchain_google_alloydb_pg.model_manager
  :members:
  :private-members:
  :noindex:
