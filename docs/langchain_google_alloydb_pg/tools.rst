Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``langchain_google_alloydb_pg`` package provides LangChain tools integrating directly with AlloyDB AI in-database machine learning functions:

- **AlloyDBSentimentTool**: Analyzes sentiment of text using ``google_ml.analyze_sentiment``. Requires AlloyDB running **PostgreSQL 17 or higher** with ``google_ml_integration``.
- **AlloyDBSummaryTool**: Summarizes text content using ``google_ml.summarize``. Requires AlloyDB running **PostgreSQL 17 or higher** with ``google_ml_integration``.
- **AlloyDBIfTool**: Evaluates natural language semantic conditions returning boolean ``True`` or ``False`` using ``google_ml.if``. Supported on AlloyDB running **PostgreSQL 14 or higher** with ``google_ml_integration``.

Database Version Requirements
-----------------------------

.. note::
   The ``google_ml.analyze_sentiment`` and ``google_ml.summarize`` functions require AlloyDB running **PostgreSQL 17 or higher**. Invoking these tools on clusters running PostgreSQL 14, 15, or 16 will raise a ``feature_not_supported`` exception from PostgreSQL.
   The ``google_ml.if`` function is available on AlloyDB PostgreSQL 14 and higher.

Custom Models and Instructions
------------------------------

All tools support specifying an optional registered model ID via the ``model_id`` parameter:
- Set ``model_id`` at tool initialization: ``AlloyDBSentimentTool(engine=engine, model_id="custom-model")``
- Or pass ``model_id`` during tool execution: ``tool.invoke({"content": "...", "model_id": "custom-model"})``

``AlloyDBSummaryTool`` also accepts ``additional_instructions`` to customize summary format and style:
- Set ``additional_instructions`` at tool initialization or pass during invocation: ``tool.invoke({"content": "...", "additional_instructions": "In 3 bullet points"})``

.. automodule:: langchain_google_alloydb_pg.tools
  :members:
  :private-members:
  :noindex:
