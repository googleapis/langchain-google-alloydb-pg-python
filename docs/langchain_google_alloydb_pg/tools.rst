Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``langchain_google_alloydb_pg`` package provides LangChain tools that call AlloyDB AI functions directly in the database:

- **AlloyDBSentimentTool**: Analyzes the sentiment of text using ``google_ml.analyze_sentiment``.
- **AlloyDBSummaryTool**: Summarizes text using ``google_ml.summarize``.

.. note::
   ``google_ml.analyze_sentiment`` and ``google_ml.summarize`` require AlloyDB running **PostgreSQL 17 or higher** with ``google_ml_integration``, and the ``google_ml_integration.enable_ai_query_engine`` and ``google_ml_integration.enable_preview_ai_functions`` database flags turned on.

The tools are standard LangChain tools. Call them with ``invoke`` / ``ainvoke``, or pass them to an agent:

.. code-block:: python

    from langchain_google_alloydb_pg import AlloyDBSentimentTool

    sentiment = AlloyDBSentimentTool(engine=engine, model_id="gemini-2.5-flash")
    sentiment.invoke({"content": "I love this product!"})  # "positive"

``model_id`` is optional and is set when you create the tool, so the agent can't change it. Without it, the database uses the model named by its ``google_ml_integration.default_llm_model`` flag.

``AlloyDBSummaryTool`` also accepts ``additional_instructions`` to control the style and format of the summary, either at initialization or per call by adding ``"additional_instructions": "In 3 bullet points"`` to the tool input.

If the AI function returns NULL, for example because the model's reply isn't one of the expected answers, the tool raises ``AlloyDBToolError``. To pass that message to the agent instead, create the tool with ``handle_tool_error=True``. Database errors, such as an unknown model, always raise.

.. automodule:: langchain_google_alloydb_pg.tools
  :members:
  :private-members:
  :noindex:
