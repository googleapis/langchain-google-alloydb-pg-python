from typing import Sequence

import pytest
from sqlalchemy import RowMapping

from langchain_google_alloydb_pg import ExplainMixin


class MyClass(ExplainMixin):
    async def query_collection(self, param1: str) -> Sequence[RowMapping]:
        if self._explain_enabled:
            return [{"result": "some_plan"}]
        return [{"result": "some_data"}]


@pytest.mark.asyncio
async def test_explain():
    instance = MyClass()
    result = await instance.explain(instance.query_collection, "test_value")
    assert result == [{"result": "some_plan"}]


@pytest.mark.asyncio
async def test_explain_no_explain():
    instance = MyClass()
    result = await instance.query_collection("test_value")
    assert result == [{"result": "some_data"}]
