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

from typing import Any, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from .engine import AlloyDBEngine


class AlloyDBModel:
    def __init__(
        self,
        model_id: str,
        model_request_url: Optional[str],
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        model_auth_type: Optional[str],
        model_auth_id: Optional[str],
        input_transform_fn: Optional[str],
        output_transform_fn: Optional[str],
        generate_headers_fn: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.model_request_url = model_request_url
        self.model_provider = model_provider
        self.model_type = model_type
        self.model_qualified_name = model_qualified_name
        self.model_auth_type = model_auth_type
        self.model_auth_id = model_auth_id
        self.input_transform_fn = input_transform_fn
        self.output_transform_fn = output_transform_fn
        # List models is returning column name "header_gen_fn"
        self.generate_headers_fn = generate_headers_fn or kwargs.get("header_gen_fn")


class AlloyDBModelManager:
    """Manage models to be used with google_ml_integration Extension.
    Refer to [Model Endpoint Management](https://cloud.google.com/alloydb/docs/ai/model-endpoint-overview).
    """

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
    ):
        """AlloyDBModelManager constructor.
        Args:
            engine (AlloyDBEngine): Connection pool engine for managing connections to Postgres database.
        """
        if key != AlloyDBModelManager.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self._engine = engine

    @classmethod
    async def create(
        cls: type[AlloyDBModelManager],
        engine: AlloyDBEngine,
    ) -> AlloyDBModelManager:
        manager = AlloyDBModelManager(cls.__create_key, engine)
        coro = manager.__avalidate()
        await engine._run_as_async(coro)
        return manager

    @classmethod
    def create_sync(
        cls: type[AlloyDBModelManager],
        engine: AlloyDBEngine,
    ) -> AlloyDBModelManager:
        manager = AlloyDBModelManager(cls.__create_key, engine)
        coro = manager.__avalidate()
        engine._run_as_sync(coro)
        return manager

    async def aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Lists the model details for a specific model_id.

        Args:
            model_id (str): A unique ID for the model endpoint that you have defined.

        Returns:
            :class: `AlloyDBModel` object of the specified model if it exists otherwise `None`.

        """
        result = await self._engine._run_as_async(self.__aget_model(model_id=model_id))
        return result

    async def alist_models(self) -> list[AlloyDBModel]:
        """Lists all the models and its details.

        Returns:
            list[`AlloyDBModel`] of all available model..
        """
        results = await self._engine._run_as_async(self.__alist_models())
        return results

    async def acreate_model(
        self,
        model_id: str,
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        **kwargs: dict[str, str],
    ) -> None:
        """Creates a registration for custom text model.

        Args:
            model_id (str): A unique ID for the model endpoint that you define.
            model_provider (str): The provider of the model endpoint.
            model_type (str): The model type. Either text_embedding or generic.
            model_qualified_name (str): The fully qualified name in case the model endpoint has multiple versions
            **kwargs :
                model_request_url (str): The model-specific endpoint when adding other text embedding and generic model endpoints
                model_auth_type (str): The authentication type used by the model endpoint.
                model_auth_id (str): The secret ID that you set and is subsequently used when registering a model endpoint.
                generate_headers_fn (str): 	The SQL function name you set to generate custom headers.
                model_in_transform_fn (str): The SQL function name to transform input of the corresponding prediction function to the model-specific input.
                model_out_transform_fn (str): The SQL function name to transform model specific output to the prediction function output.

        Returns:
          None

        Raises:
            :class:`DBAPIError <sqlalchemy.exc.DBAPIError>`: if argument names mismatch create_model function specification.
        """
        await self._engine._run_as_async(
            self.__acreate_model(
                model_id,
                model_provider,
                model_type,
                model_qualified_name,
                **kwargs,
            )
        )

    async def adrop_model(self, model_id: str) -> None:
        """Removes an already registered model.

        Args:
            model_id (str): A unique ID for the model endpoint that you have defined.

        Returns:
            None
        """
        await self._engine._run_as_async(self.__adrop_model(model_id))

    async def __avalidate(self) -> None:
        """Private async function to validate prerequisites.

        Raises:
            Exception if google_ml_integration EXTENSION is not 1.3.
            Exception if google_ml_integration.enable_model_support DB Flag not set.
        """
        extension_version = await self.__fetch_google_ml_extension()
        db_flag = await self.__fetch_db_flag()
        if extension_version < "1.3":
            raise Exception(
                "Please upgrade google_ml_integration EXTENSION to version 1.3 or above."
            )
        if db_flag != "on":
            raise Exception(
                "google_ml_integration.enable_model_support DB Flag not set."
            )

    async def __query_db(self, query: str) -> Sequence[RowMapping]:
        """Queries the Postgres database through the engine.

        Args:
            query (str): Query to execute on the DB.

        Raises:
            Exception if the query is not a returning type."""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return results

    async def __aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Lists the model details for a specific model_id. Returns None if it doesn't exist.

        Args:
            model_id (str): A unique ID for the model endpoint that you have defined.

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
                generate_headers_fn VARCHAR,
                input_transform_fn VARCHAR,
                output_transform_fn VARCHAR)"""

        try:
            result = await self.__query_db(query)
        except Exception:
            return None
        data_class = self.__convert_dict_to_dataclass(result)[0]
        return data_class

    async def __alist_models(self) -> list[AlloyDBModel]:
        """Lists all the models and its details."""
        query = "SELECT * FROM google_ml.model_info_view;"
        result = await self.__query_db(query)
        list_of_data_classes = self.__convert_dict_to_dataclass(result)
        return list_of_data_classes

    async def __acreate_model(
        self,
        model_id: str,
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        **kwargs: dict[str, str],
    ) -> None:
        """Creates a registration for custom text model.

        Args:
            model_id (str): A unique ID for the model endpoint that you define.
            model_provider (str): The provider of the model endpoint.
            model_type (str): The model type. Either text_embedding or generic.
            model_qualified_name (str): The fully qualified name in case the model endpoint has multiple versions.
            **kwargs :
                model_request_url (str): The model-specific endpoint when adding other text embedding and generic model endpoints
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
        model_provider => '{model_provider}',
        model_type => '{model_type}',
        model_qualified_name => '{model_qualified_name}',"""
        for key, value in kwargs.items():
            query = query + f" {key} => '{value}',"
        query = query.strip(",")
        query = query + ");"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def __adrop_model(self, model_id: str) -> None:
        """Removes an already registered model.

        Args:
            model_id (str): A unique ID for the model endpoint that you have defined.
        """
        query = f"CALL google_ml.drop_model('{model_id}');"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def __fetch_google_ml_extension(self) -> str:
        """Creates the Google ML Extension if it does not exist and returns the version number (Default creates version 1.3)."""
        create_extension_query = """
        DO $$
        BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_extension WHERE extname = 'google_ml_integration' )
          THEN CREATE EXTENSION google_ml_integration VERSION '1.3' CASCADE;
        END IF;
        END
        $$;
        """
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(create_extension_query))
            await conn.commit()
        extension_version_query = "SELECT extversion FROM pg_extension WHERE extname = 'google_ml_integration';"
        result = await self.__query_db(extension_version_query)
        version = result[0]["extversion"]
        return version

    async def __fetch_db_flag(self) -> str:
        """Fetches the enable_model_support DB flag."""
        db_flag_query = "SELECT setting FROM pg_settings where name = 'google_ml_integration.enable_model_support';"
        result = await self.__query_db(db_flag_query)
        flag = result[0]["setting"]
        return flag

    def __convert_dict_to_dataclass(
        self, list_of_rows: Sequence[RowMapping]
    ) -> list[AlloyDBModel]:
        """Converts a list of DB rows to list of AlloyDBModel dataclass.

        Args:
            list_of_rows (Sequence[RowMapping]): A unique ID for the model endpoint that you define.
        """
        list_of_dataclass = [AlloyDBModel(**row) for row in list_of_rows]
        return list_of_dataclass
