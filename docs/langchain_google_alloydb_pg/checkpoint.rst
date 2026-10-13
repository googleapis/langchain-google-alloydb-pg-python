Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBSaver`` is a LangGraph checkpointer (``BaseCheckpointSaver``) that persists graph state in AlloyDB tables, so agents can pause, resume, and recover state across conversation threads.

Initialize the checkpoint tables once with ``AlloyDBEngine.init_checkpoint_table`` (or ``ainit_checkpoint_table``), then pass the saver to ``compile``:

.. code-block:: python

    from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBSaver

    engine = AlloyDBEngine.from_instance(
        "project-id", "region", "my-cluster", "my-instance", "my-database"
    )
    engine.init_checkpoint_table()  # once per database

    checkpointer = AlloyDBSaver.create_sync(engine)
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-123"}}
    graph.invoke({"messages": [("user", "Hello")]}, config)

See the full `LangGraph Checkpoint <https://github.com/googleapis/langchain-google-alloydb-pg-python/tree/main/docs/langgraph_checkpoint.ipynb>`_ tutorial.

.. automodule:: langchain_google_alloydb_pg.checkpoint
  :members:
  :private-members:
  :noindex:
