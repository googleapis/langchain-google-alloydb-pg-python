"""Global pytest configuration for langchain-google-alloydb-pg-python."""

import os
import pytest

def pytest_collection_modifyitems(config, items):
    """Skip tests that require GCP environment variables if PROJECT_ID is not set."""
    if os.environ.get("PROJECT_ID"):
        return

    skip_gcp = pytest.mark.skip(reason="Missing required GCP environment variables")
    
    # List of test files/modules that require live GCP / AlloyDB connection
    gcp_test_files = {
        "test_async_chatmessagehistory.py",
        "test_async_checkpoint.py",
        "test_async_loader.py",
        "test_async_vectorstore.py",
        "test_async_vectorstore_from_methods.py",
        "test_async_vectorstore_index.py",
        "test_async_vectorstore_search.py",
        "test_chatmessagehistory.py",
        "test_checkpoint.py",
        "test_embeddings.py",
        "test_engine.py",
        "test_loader.py",
        "test_model_manager.py",
        "test_pgvector_migrator.py",
        "test_standard_test_suite.py",
        "test_vectorstore.py",
        "test_vectorstore_embeddings.py",
        "test_vectorstore_from_methods.py",
        "test_vectorstore_index.py",
        "test_vectorstore_search.py",
    }

    for item in items:
        # Check if the test is from one of the GCP test files (supporting pytest 8/9 item.path)
        if hasattr(item, "path"):
            fspath_name = item.path.name
        else:
            fspath_name = getattr(getattr(item, "fspath", None), "basename", "")
        if fspath_name in gcp_test_files:
            item.add_marker(skip_gcp)
        # Also skip tests with 'integration' or 'live' in their name/nodeid, or requiring 'engine' fixture
        nodeid = getattr(item, "nodeid", "")
        if (
            "integration" in nodeid.lower()
            or "live" in nodeid.lower()
        ):
            item.add_marker(skip_gcp)
        elif "engine" in getattr(item, "fixturenames", []):
            fixtureinfo = getattr(item, "_fixtureinfo", None)
            is_local_fixture = False
            if fixtureinfo:
                fixturedefs = fixtureinfo.name2fixturedefs.get("engine", [])
                if fixturedefs and "conftest" not in fixturedefs[-1].func.__module__:
                    is_local_fixture = True
            
            if not is_local_fixture:
                item.add_marker(skip_gcp)

