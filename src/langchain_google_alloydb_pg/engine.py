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
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import aiohttp
import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
from google.cloud.alloydb.connector import (
    AsyncConnector,
    IPTypes,
    RefreshStrategy,
)
from langchain_postgres import Column, PGEngine
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import create_async_engine

from .version import __version__

if TYPE_CHECKING:
    import asyncpg  # type: ignore
    import google.auth.credentials  # type: ignore

T = TypeVar("T")

USER_AGENT = "langchain-google-alloydb-pg-python/" + __version__

CHECKPOINTS_TABLE = "checkpoints"


async def _get_iam_principal_email(
    credentials: google.auth.credentials.Credentials,
) -> str:
    """Get email address associated with current authenticated IAM principal.

    Email will be used for automatic IAM database authentication to AlloyDB.

    Args:
        credentials (google.auth.credentials.Credentials):
            The credentials object to use in finding the associated IAM
            principal email address.

    Returns:
        email (str):
            The email address associated with the current authenticated IAM
            principal.
    """
    # refresh credentials if they are not valid
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    if hasattr(credentials, "_service_account_email"):
        return credentials._service_account_email.replace(".gserviceaccount.com", "")
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url, raise_for_status=True)
        response_json: dict = await response.json()
        email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM principal's "
            "email address using environment's ADC credentials!"
        )
    return email.replace(".gserviceaccount.com", "")


class AlloyDBEngine(PGEngine):
    """A class for managing connections to a AlloyDB database."""

    _connector: Optional[AsyncConnector] = None

    @classmethod
    def __start_background_loop(
        cls,
        project_id: str,
        region: str,
        cluster: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> Future:
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()
        coro = cls._create(
            project_id,
            region,
            cluster,
            instance,
            database,
            ip_type,
            user,
            password,
            loop=cls._default_loop,
            thread=cls._default_thread,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return asyncio.run_coroutine_threadsafe(coro, cls._default_loop)

    @classmethod
    def from_instance(
        cls: type[AlloyDBEngine],
        project_id: str,
        region: str,
        cluster: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine from an AlloyDB instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Cloud AlloyDB instance region.
            cluster (str): Cloud AlloyDB cluster name.
            instance (str): Cloud AlloyDB instance name.
            database (str): Database name.
            user (Optional[str]): Cloud AlloyDB user name. Defaults to None.
            password (Optional[str]): Cloud AlloyDB user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Returns:
            AlloyDBEngine: A newly created AlloyDBEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            cluster,
            instance,
            database,
            user,
            password,
            ip_type,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return future.result()

    @classmethod
    async def _create(
        cls: type[AlloyDBEngine],
        project_id: str,
        region: str,
        cluster: str,
        instance: str,
        database: str,
        ip_type: Union[str, IPTypes],
        user: Optional[str] = None,
        password: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine from an AlloyDB instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Cloud AlloyDB instance region.
            cluster (str): Cloud AlloyDB cluster name.
            instance (str): Cloud AlloyDB instance name.
            database (str): Database name.
            ip_type (Union[str, IPTypes]): IP address type. Defaults to IPTypes.PUBLIC.
            user (Optional[str]): Cloud AlloyDB user name. Defaults to None.
            password (Optional[str]): Cloud AlloyDB user password. Defaults to None.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.
            iam_account_email (Optional[str]): IAM service account email.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Raises:
            ValueError: Raises error if only one of 'user' or 'password' is specified.

        Returns:
            AlloyDBEngine: A newly created AlloyDBEngine instance.
        """
        # error if only one of user or password is set, must be both or neither
        if bool(user) ^ bool(password):
            raise ValueError(
                "Only one of 'user' or 'password' were specified. Either "
                "both should be specified to use basic user/password "
                "authentication or neither for IAM DB authentication."
            )

        if cls._connector is None:
            cls._connector = AsyncConnector(
                user_agent=USER_AGENT, refresh_strategy=RefreshStrategy.LAZY
            )

        # if user and password are given, use basic auth
        if user and password:
            enable_iam_auth = False
            db_user = user
        # otherwise use automatic IAM database authentication
        else:
            enable_iam_auth = True
            if iam_account_email:
                db_user = iam_account_email
            else:
                # get application default credentials
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/userinfo.email"]
                )
                db_user = await _get_iam_principal_email(credentials)

        # anonymous function to be used for SQLAlchemy 'creator' argument
        async def getconn() -> asyncpg.Connection:
            conn = await cls._connector.connect(  # type: ignore
                f"projects/{project_id}/locations/{region}/clusters/{cluster}/instances/{instance}",
                "asyncpg",
                user=db_user,
                password=password,
                db=database,
                enable_iam_auth=enable_iam_auth,
                ip_type=ip_type,
            )
            return conn

        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
            **engine_args,
        )
        return cls(PGEngine._PGEngine__create_key, engine, loop, thread)  # type: ignore

    @classmethod
    async def afrom_instance(
        cls: type[AlloyDBEngine],
        project_id: str,
        region: str,
        cluster: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine from an AlloyDB instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Cloud AlloyDB instance region.
            cluster (str): Cloud AlloyDB cluster name.
            instance (str): Cloud AlloyDB instance name.
            database (str): Cloud AlloyDB database name.
            user (Optional[str], optional): Cloud AlloyDB user name. Defaults to None.
            password (Optional[str], optional): Cloud AlloyDB user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Returns:
            AlloyDBEngine: A newly created AlloyDBEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            cluster,
            instance,
            database,
            user,
            password,
            ip_type,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return await asyncio.wrap_future(future)

    @classmethod
    def from_connection_string(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine instance from arguments
        Args:
            url (Optional[str]): the URL used to connect to a database. Use url or set other arguments.
        Raises:
            ValueError: If not all database url arguments are specified
        Returns:
            AlloyDBEngine
        """

        return AlloyDBEngine.from_engine_args(url=url, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine instance from arguments

        Args:
            url (Optional[str]): the URL used to connect to a database. Use url or set other arguments.

        Raises:
            ValueError: If not all database url arguments are specified

        Returns:
            AlloyDBEngine
        """
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        driver = "postgresql+asyncpg"
        if (isinstance(url, str) and not url.startswith(driver)) or (
            isinstance(url, URL) and url.drivername != driver
        ):
            raise ValueError("Driver must be type 'postgresql+asyncpg'")

        engine = create_async_engine(url, **kwargs)
        return cls(PGEngine._PGEngine__create_key, engine, cls._default_loop, cls._default_thread)  # type: ignore

    async def _ainit_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """
        Create an AlloyDB table to save chat history messages.

        Args:
            table_name (str): The table name to store chat history.
            schema_name (str): The schema name to store the chat history table.
                Default: "public".

        Returns:
            None
        """
        create_table_query = f"""CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}"(
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            data JSONB NOT NULL,
            type TEXT NOT NULL
        );"""
        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.commit()

    async def ainit_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """Create an AlloyDB table to save chat history messages.

        Args:
            table_name (str): The table name to store chat history.
            schema_name (str): The schema name to store chat history table.
                Default: "public".

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_chat_history_table(
                table_name,
                schema_name,
            )
        )

    def init_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """Create a Cloud SQL table to store chat history.

        Args:
            table_name (str): Table name to store chat history.
            schema_name (str): The schema name to store chat history table.
                Default: "public".

        Returns:
            None
        """
        self._run_as_sync(self._ainit_chat_history_table(table_name, schema_name))

    async def _ainit_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.
        If table already exists, a DuplicateTableError error is thrown.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (list[Column]): A list of Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).
        """
        query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            {content_column} TEXT NOT NULL
            """
        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            query += f',\n"{column.name}" {column.data_type} {nullable}'
        metadata_json_column = metadata_json_column or "langchain_metadata"
        if store_metadata:
            query += f',\n"{metadata_json_column}" JSON'
        query += "\n);"

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def ainit_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (list[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
        """
        await self._run_as_async(
            self._ainit_document_table(
                table_name,
                schema_name,
                content_column,
                metadata_columns,
                metadata_json_column,
                store_metadata,
            )
        )

    def init_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name to store the PgSQL database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (list[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
        """
        self._run_as_sync(
            self._ainit_document_table(
                table_name,
                schema_name,
                content_column,
                metadata_columns,
                metadata_json_column,
                store_metadata,
            )
        )

    async def _ainit_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """
        Create AlloyDB tables to save checkpoints.

        Args:
            schema_name (str): The schema name to store the checkpoint tables.
                Default: "public".

        Returns:
            None
        """
        create_checkpoints_table = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BYTEA,
            metadata BYTEA,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );"""

        create_checkpoint_writes_table = f"""CREATE TABLE "{schema_name}"."{table_name + "_writes"}"(
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BYTEA NOT NULL,
            task_path TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );"""

        async with self._pool.connect() as conn:
            await conn.execute(text(create_checkpoints_table))
            await conn.execute(text(create_checkpoint_writes_table))
            await conn.commit()

    async def ainit_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """Create an AlloyDB table to save checkpoint messages.

        Args:
            schema_name (str): The schema name to store checkpoint tables.
                Default: "public".

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_checkpoint_table(
                table_name,
                schema_name,
            )
        )

    def init_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """Create Cloud SQL tables to store checkpoints.

        Args:
            schema_name (str): The schema name to store checkpoint tables.
                Default: "public".

        Returns:
            None
        """
        self._run_as_sync(self._ainit_checkpoint_table(table_name, schema_name))

    async def _aload_table_schema(
        self, table_name: str, schema_name: str = "public"
    ) -> Table:
        """
        Load table schema from an existing table in a PgSQL database, potentially from a specific database schema.

        Args:
            table_name: The name of the table to load the table schema from.
            schema_name: The name of the database schema where the table resides.
                Default: "public".

        Returns:
            (sqlalchemy.Table): The loaded table, including its table schema information.
        """
        metadata = MetaData()
        async with self._pool.connect() as conn:
            try:
                await conn.run_sync(
                    metadata.reflect, schema=schema_name, only=[table_name]
                )
            except InvalidRequestError as e:
                raise ValueError(
                    f"Table, '{schema_name}'.'{table_name}', does not exist: " + str(e)
                )

        table = Table(table_name, metadata, schema=schema_name)
        # Extract the schema information
        schema = []
        for column in table.columns:
            schema.append(
                {
                    "name": column.name,
                    "type": column.type.python_type,
                    "max_length": getattr(column.type, "length", None),
                    "nullable": not column.nullable,
                }
            )

        return metadata.tables[f"{schema_name}.{table_name}"]
