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

from dataclasses import dataclass
from typing import Any, List, Sequence

from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class AlloyDBModel:
    model_id: str
    model_request_url: str
    model_provider: str
    model_type: str
    model_qualified_name: str
    model_auth_type: str
    model_auth_id: str
    input_transform_fn: str
    output_transform_fn: str


class AlloyDBModelManager:
    """Manage models to be used with google_ml_integration Extension.
    Refer to [Model Endpoint Management](https://cloud.google.com/alloydb/docs/ai/model-endpoint-overview).
    """

    def __init__(
        self,
        engine: AsyncEngine,
    ):
        """AlloyDBModelManager constructor.
        Args:
            engine (Asycn AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
        """

        self._engine = engine

        self._engine._run_as_sync(self.__validate())

    async def alist_model(
        self, model_id: str = "textembedding-gecko@003"
    ) -> AlloyDBModel:
        """Lists the model details for a specific model_id.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if model has not been created.
        """
        result = await self._engine._run_as_async(self.__alist_model(model_id=model_id))
        return result

    async def amodel_info_view(self) -> List[AlloyDBModel]:
        """Lists all the models and its details."""
        results = await self._engine._run_as_async(self.__amodel_info_view())
        return results

    async def acreate_model(
        self, model_id: str, model_provider: str, **kwargs: dict[str, Any]
    ) -> None:```
        """Creates a custom text embedding model.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if argument names mismatch create_model function specification.
        """
        await self._engine._run_as_async(
            self.__acreate_model(model_id, model_provider, **kwargs)
        )

    async def adrop_model(self, model_id: str) -> None:
        """Removes a text embedding model."""
        await self._engine._run_as_async(self.__adrop_model(model_id))

    async def __validate(self) -> None:
        """Private function to validate prerequisites.

        Raises:
            Exception if google_ml_integration EXTENSION is not 1.3.
            Exception if google_ml_integration.enable_model_support DB Flag not set.
        """
        extension_version = await self.__fetch_google_ml_extension()
        db_flag = await self.__fetch_db_flag()
        if extension_version < 1.3:
            raise Exception("google_ml_integration EXTENSION is not 1.3")
        if db_flag != "on":
            raise Exception(
                "google_ml_integration.enable_model_support DB Flag not set."
            )

    async def __query_db(self, query: str) -> Sequence[RowMapping]:
        """Queries the Postgres database through the engine.

        Raises:
            Exception if the query is not a returning type."""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return results

    async def __alist_model(
        self,
        model_id: str = "textembedding-gecko@001",
    ) -> AlloyDBModel:
        """Lists the model details for a specific model_id.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if model has not been created.
        """
        query = f"""SELECT * FROM
                google_ml.list_model('{model_id}')
                AS t(model_id VARCHAR,
                model_request_url VARCHAR,
                model_provider google_ml.model_provider,
                model_type google_ml.model_type,
                model_qualified_name VARCHAR,
                model_auth_type google_ml.auth_type,
                model_auth_id VARCHAR,
                input_transform_fn VARCHAR,
                output_transform_fn VARCHAR)"""

        result = await self.__query_db(query)
        data_class = self.__convert_dict_to_dataclass(result)[0]
        return data_class

    async def __amodel_info_view(self) -> List[AlloyDBModel]:
        """Lists all the models and its details."""
        query = "SELECT * FROM google_ml.model_info_view;"
        result = await self.__query_db(query)
        list_of_data_classes = self.__convert_dict_to_dataclass(result)
        return list_of_data_classes

    async def __acreate_model(
        self, model_id: str, model_provider: str, **kwargs: dict[str, Any]
    ) -> None:
        """Creates a custom text embedding model.

        Args:
            model_id (str): A unique ID for the model endpoint that you define.
            model_provider (str): The provider of the model endpoint.
            **kwargs :
                model_request_url (str): The model-specific endpoint when adding other text embedding and generic model endpoints
                model_type (str): The model type. Either text_embedding or generic.
                model_qualified_name (str): The fully qualified name in case the model endpoint has multiple versions
                model_auth_type (str): The authentication type used by the model endpoint.
                model_auth_id (str): The secret ID that you set and is subsequently used when registering a model endpoint.
                generate_headers_fn (str): 	The SQL function name you set to generate custom headers.
                model_in_transform_fn (str): The SQL function name to transform input of the corresponding prediction function to the model-specific input.
                model_out_transform_fn (str): The SQL function name to transform model specific output to the prediction function output.

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if argument names mismatch create_model function specification.
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

    async def __adrop_model(self, model_id: str) -> None:
        """Removes a text embedding model."""
        query = f"CALL google_ml.drop_model('{model_id}');"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def __fetch_google_ml_extension(self) -> float:
        """Fetches the version of the Google ML Extension."""
        extension_query = """select coalesce((select extversion FROM pg_extension where extname = 'google_ml_integration'), '0') as extversion;"""
        result = await self.__query_db(extension_query)
        version = result[0]["extversion"]
        return float(version)

    async def __fetch_db_flag(self) -> str:
        """Fetches the enable_model_support DB flag."""
        db_flag_query = "SELECT setting FROM pg_settings where name = 'google_ml_integration.enable_model_support';"
        result = await self.__query_db(db_flag_query)
        flag = result[0]["setting"]
        return flag

    def __convert_dict_to_dataclass(self, list_of_rows):
        list_of_dataclass = [AlloyDBModel(**row) for row in list_of_rows]
        return list_of_dataclass
