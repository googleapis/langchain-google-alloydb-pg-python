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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

from typing import List, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class AlloyDBModel:
    """Text embedding models to be used with google_ml_integration."""

    def __init__(
        self,
        engine: AsyncEngine,
    ):
        """AlloyDBModel constructor.
        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
        """

        self._engine = engine

        self._validate()

    async def validateGoogleMLExtension(self) -> None:
        """Validates the version compatibility of the Google ML Extension.

        Raises:
            Assertion error if google_ml_integration EXTENSION is not 1.3.
        """
        extension_query = (
            "SELECT * FROM pg_extension where extname = 'google_ml_integration';"
        )
        result = await self._query_db(extension_query)
        assert (
            float(result[0]["extversion"]) == 1.3
        ), "google_ml_integration EXTENSION is not 1.3"

    async def validateDBFlag(self) -> None:
        """Validates if the enable_model_support flag is set.

        Raises:
            Assertion error if google_ml_integration.enable_model_support DB Flag not set.
        """
        db_flag_query = "SELECT name, setting FROM pg_settings where name = 'google_ml_integration.enable_model_support';"
        result = await self._query_db(db_flag_query)
        assert (
            result[0]["setting"] == "on"
        ), "google_ml_integration.enable_model_support DB Flag not set"

    async def alist_model(
        self, model_id: str = "textembedding-gecko@001"
    ) -> List[Sequence]:
        """Lists the model details for a specific model_id.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if model has not been created.
        """
        result = await self._engine._run_as_async(self._alist_model(model_id=model_id))
        return result

    async def amodel_info_view(self) -> List[Sequence]:
        """Lists all the models and its details."""
        results = await self._engine._run_as_async(self._amodel_info_view())
        return results

    async def acreate_model(self, model_id: str, model_provider: str, **kwargs) -> None:
        """Creates a custom text embedding model.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if argument names are not the same as those in the SQL function.
        """
        await self._engine._run_as_async(
            self._acreate_model(model_id, model_provider, **kwargs)
        )

    async def adrop_model(self, model_id: str) -> None:
        """Removes a text embedding model."""
        await self._engine._run_as_async(self._adrop_model(model_id))

    def _validate(self) -> None:
        """Private function to validate prerequisites.

        Raises:
            Assertion error if google_ml_integration EXTENSION is not 1.3.
            Assertion error if google_ml_integration.enable_model_support DB Flag not set.
        """
        self._engine._run_as_sync(self.validateGoogleMLExtension())
        self._engine._run_as_sync(self.validateDBFlag())

    async def _query_db(self, query: str) -> List[Sequence]:
        """Queries the Postgres database through the engine.

        Raises:
            Exception if the query is not a returning type."""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return results

    async def _alist_model(
        self,
        model_id: str = "textembedding-gecko@001",
    ) -> List[Sequence]:
        """Lists the model details for a specific model_id.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if model has not been created.
        """
        query = f"""SELECT * from google_ml.list_model('{model_id}')as t(model_id VARCHAR, model_request_url VARCHAR,
        model_provider google_ml.model_provider, model_type google_ml.model_type, model_qualified_name VARCHAR ,
        model_auth_type google_ml.auth_type, model_auth_id VARCHAR, input_transform_fn VARCHAR,output_transform_fn VARCHAR)"""
        result = await self._query_db(query)
        return result

    async def _amodel_info_view(self) -> List[Sequence]:
        """Lists all the models and its details."""
        query = "SELECT * FROM google_ml.model_info_view;"
        result = await self._query_db(query)
        return result

    async def _acreate_model(
        self, model_id: str, model_provider: str, **kwargs
    ) -> None:
        """Creates a custom text embedding model.

        Args:
            model_id (str): A unique ID for the model endpoint that you define.
            model_provider (str): The provider of the model endpoint.
            **kawrgs :
                model_request_url (str): The model-specific endpoint when adding other text embedding and generic model endpoints
                model_type (str): The model type. Either text_embedding or generic.
                model_qualified_name (str): The fully qualified name in case the model endpoint has multiple versions
                model_auth_type (str): The authentication type used by the model endpoint.
                model_auth_id (str): The secret ID that you set and is subsequently used when registering a model endpoint.
                generate_headers_fn (str): 	The SQL function name you set to generate custom headers.
                model_in_transform_fn (str): The SQL function name to transform input of the corresponding prediction function to the model-specific input.
                model_out_transform_fn (str): The SQL function name to transform model specific output to the prediction function output.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if argument names are not the same as those in the SQL function.
        """
        query = f"""
        CALL
        google_ml.create_model(
        model_id => '{model_id}',
        model_provider => '{model_provider}',"""
        for key, value in kwargs.items():
            query = query + f" {key} => '{value}',"
        query = query.strip(",")
        query = query + ");"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def _adrop_model(self, model_id: str) -> None:
        """Removes a text embedding model."""
        query = f"CALL google_ml.drop_model('{model_id}');"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()
