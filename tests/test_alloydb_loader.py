# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import uuid

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBLoader

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
cluster = os.environ["CLUSTER_ID"]
table_name = "test-table" + str(uuid.uuid4())


@pytest.mark.asyncio
class TestAlloyDBLoaderAsync:
    @pytest_asyncio.fixture
    async def engine(self):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=project_id,
            instance=instance_id,
            region=region,
            cluster=cluster,
            database=db_name,
        )
        yield engine

    async def _collect_async_items(self, docs_generator):
        """Collects items from an async generator."""
        return [doc async for doc in docs_generator]

    async def _cleanup_table(self, engine):
        query = f'DROP TABLE IF EXISTS "{table_name}"'
        await engine._aexecute(query)

    async def test_load_from_query_default(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                INSERT INTO "{table_name}" (
                    fruit_name, variety, quantity_in_stock, price_per_unit, organic
                ) VALUES ('Apple', 'Granny Smith', 150, 1, 1);
            """
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                table_name=table_name,
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="1",
                    metadata={
                        "fruit_name": "Apple",
                        "variety": "Granny Smith",
                        "quantity_in_stock": 150,
                        "price_per_unit": 1,
                        "organic": 1,
                    },
                )
            ]
        except Exception as e:
            print(e)
        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_customized_content_customized_metadata(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                INSERT INTO "{table_name}" (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES ('Apple', 'Granny Smith', 150, 0.99, 1),
                       ('Banana', 'Cavendish', 200, 0.59, 0),
                       ('Orange', 'Navel', 80, 1.29, 1);
            """
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                content_columns=[
                    "fruit_name",
                    "variety",
                    "quantity_in_stock",
                    "price_per_unit",
                    "organic",
                ],
                metadata_columns=["fruit_id"],
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="Apple Granny Smith 150 1 1",
                    metadata={"fruit_id": 1},
                ),
                Document(
                    page_content="Banana Cavendish 200 1 0",
                    metadata={"fruit_id": 2},
                ),
                Document(
                    page_content="Orange Navel 80 1 1",
                    metadata={"fruit_id": 3},
                ),
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_customized_content_default_metadata(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                INSERT INTO "{table_name}" (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES ('Apple', 'Granny Smith', 150, 1, 1);
            """
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                content_columns=[
                    "variety",
                    "quantity_in_stock",
                    "price_per_unit",
                ],
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="Granny Smith 150 1",
                    metadata={
                        "fruit_id": 1,
                        "fruit_name": "Apple",
                        "organic": 1,
                    },
                )
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_default_content_customized_metadata(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                        INSERT INTO "{table_name}" (
                            fruit_name,
                            variety,
                            quantity_in_stock,
                            price_per_unit,
                            organic
                        ) VALUES ('Apple', 'Granny Smith', 150, 1, 1);
            """
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                metadata_columns=["fruit_name", "organic"],
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="1",
                    metadata={"fruit_name": "Apple", "organic": 1},
                )
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_with_langchain_metadata(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                CREATE TABLE IF NOT EXISTS "{table_name}"(
                    fruit_id SERIAL PRIMARY KEY,
                    fruit_name VARCHAR(100) NOT NULL,
                    variety VARCHAR(50),
                    quantity_in_stock INT NOT NULL,
                    price_per_unit INT NOT NULL,
                    langchain_metadata JSON NOT NULL
                )
                """
            await engine._aexecute(query)

            metadata = json.dumps({"organic": 1})
            insert_query = f"""
                INSERT INTO "{table_name}"
                (fruit_name, variety, quantity_in_stock, price_per_unit, langchain_metadata)
                VALUES ('Apple', 'Granny Smith', 150, 1, '{metadata}');"""
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                metadata_columns=[
                    "fruit_name",
                    "langchain_metadata",
                ],
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="1",
                    metadata={
                        "fruit_name": "Apple",
                        "organic": 1,
                    },
                )
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_with_json(self, engine):
        try:
            await self._cleanup_table(engine)
            query = f"""
                CREATE TABLE IF NOT EXISTS "{table_name}"(
                    fruit_id SERIAL PRIMARY KEY,
                    fruit_name VARCHAR(100) NOT NULL,
                    variety JSON NOT NULL,
                    quantity_in_stock INT NOT NULL,
                    price_per_unit INT NOT NULL,
                    langchain_metadata JSON NOT NULL
                )
                """
            await engine._aexecute(query)

            metadata = json.dumps({"organic": 1})
            variety = json.dumps({"type": "Granny Smith"})
            insert_query = f"""
                INSERT INTO "{table_name}"
                (fruit_name, variety, quantity_in_stock, price_per_unit, langchain_metadata)
                VALUES ('Apple', '{variety}', 150, 1, '{metadata}');"""
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                metadata_columns=[
                    "variety",
                ],
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="1",
                    metadata={
                        "variety": {"type": "Granny Smith"},
                        "organic": 1,
                    },
                )
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_customized_content_default_metadata_custom_formatter(
        self, engine
    ):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                        INSERT INTO "{table_name}" (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                        VALUES ('Apple', 'Granny Smith', 150, 1, 1);
                        """
            await engine._aexecute(insert_query)

            def my_formatter(row, content_columns):
                return "-".join(
                    str(row[column]) for column in content_columns if column in row
                )

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                content_columns=[
                    "variety",
                    "quantity_in_stock",
                    "price_per_unit",
                ],
                formatter=my_formatter,
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="Granny Smith-150-1",
                    metadata={
                        "fruit_id": 1,
                        "fruit_name": "Apple",
                        "organic": 1,
                    },
                )
            ]

        finally:
            await self._cleanup_table(engine)

    async def test_load_from_query_customized_content_default_metadata_custom_page_content_format(
        self, engine
    ):
        try:
            await self._cleanup_table(engine)
            query = f"""
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        fruit_id SERIAL PRIMARY KEY,
                        fruit_name VARCHAR(100) NOT NULL,
                        variety VARCHAR(50),
                        quantity_in_stock INT NOT NULL,
                        price_per_unit INT NOT NULL,
                        organic INT NOT NULL
                    )
                """
            await engine._aexecute(query)

            insert_query = f"""
                            INSERT INTO "{table_name}" (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                            VALUES ('Apple', 'Granny Smith', 150, 1, 1);
                        """
            await engine._aexecute(insert_query)

            loader = AlloyDBLoader(
                engine=engine,
                query=f'SELECT * FROM "{table_name}";',
                content_columns=[
                    "variety",
                    "quantity_in_stock",
                    "price_per_unit",
                ],
                format="YAML",
            )

            documents = await self._collect_async_items(loader.alazy_load())

            assert documents == [
                Document(
                    page_content="variety: Granny Smith\nquantity_in_stock: 150\nprice_per_unit: 1",
                    metadata={
                        "fruit_id": 1,
                        "fruit_name": "Apple",
                        "organic": 1,
                    },
                )
            ]

        finally:
            await self._cleanup_table(engine)
