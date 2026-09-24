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
from logging import getLogger
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
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from .version import __version__

if TYPE_CHECKING:
    import asyncpg  # type: ignore
    import google.auth.credentials  # type: ignore

T = TypeVar("T")

USER_AGENT = "langchain-google-alloydb-pg-python/" + __version__

CHECKPOINTS_TABLE = "checkpoints"

logger = getLogger(__name__)

# SQLSTATE codes that indicate a missing extension object.
_UNDEFINED_FUNCTION = "42883"
_INVALID_SCHEMA_NAME = "3F000"
_UNDEFINED_TABLE = "42P01"

_COLUMNAR_ENGINE_MISSING_MSG = (
    "AlloyDB Columnar Engine is not installed or enabled on this instance. "
    "Please ensure 'google_columnar_engine' is in shared_preload_libraries "
    "and 'google_columnar_engine.enabled = on' is set in instance flags."
)
_VECTOR_ASSIST_MISSING_MSG = (
    "AlloyDB Vector Assist extension is not installed on this database. "
    "Please execute 'CREATE EXTENSION IF NOT EXISTS vector_assist CASCADE;' as a superuser."
)
_AI_INITIALIZE_EMBEDDINGS_MISSING_MSG = (
    "ai.initialize_embeddings is not available on this database. It is provided "
    "by the google_ml_integration extension, version 1.5.6 or later. Please execute "
    "'CREATE EXTENSION IF NOT EXISTS google_ml_integration CASCADE;' or "
    "'ALTER EXTENSION google_ml_integration UPDATE;'."
)

# Whether a function/procedure named :name exists (in schema :schema, or in any
# schema if :schema is NULL). Used to confirm that a missing-object SQLSTATE
# really comes from the extension function, and not e.g. from a nonexistent
# user schema or a relation referenced by the call.
_FUNCTION_EXISTS_QUERY = (
    "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_proc p "
    "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
    "WHERE p.proname = :name "
    "AND n.nspname = COALESCE(CAST(:schema AS TEXT), n.nspname))"
)

# Columns added to the columnar engine when no column list is given: every
# column of the table, in table order, except pgvector-typed columns (they use
# columnar memory without speeding up analytical scans) and the hidden
# google_ml_track_stale_embedding_<n> columns that ai.initialize_embeddings
# adds. The LIKE pattern escapes "_" so it is not a single-character wildcard.
_DEFAULT_COLUMNAR_COLUMNS_QUERY = (
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_schema = :schema_name AND table_name = :table_name "
    "AND udt_name NOT IN ('vector', 'halfvec', 'sparsevec') "
    "AND column_name NOT LIKE :tracking_column_pattern "
    "ORDER BY ordinal_position"
)
_TRACKING_COLUMN_PATTERN = r"google\_ml\_track\_stale\_embedding\_%"

# Most recently defined Vector Assist spec for a table / schema / vector column.
# Shared by apply_vector_assist_spec and get_vector_assist_recommendations so
# both always resolve the same spec. created_at is the defining transaction's
# start time, so spec_id breaks ties deterministically. define_spec resolves a
# NULL schema_name from the search_path, hence the current_schema() fallback.
_LATEST_VECTOR_ASSIST_SPEC_QUERY = (
    "SELECT spec_id FROM vector_assist.vector_specs "
    "WHERE table_name = :table_name "
    "AND schema_name = COALESCE(CAST(:schema_name AS TEXT), current_schema()) "
    "AND vector_column_name = :embedding_column "
    "ORDER BY created_at DESC, spec_id DESC LIMIT 1"
)


def _quote_ident(ident: str) -> str:
    """Quote a PostgreSQL identifier (doubling embedded double quotes)."""
    return '"' + ident.replace('"', '""') + '"'


def _is_missing_object_error(error: BaseException, sqlstates: set[str]) -> bool:
    """Return True if the DB-API error behind ``error`` has one of ``sqlstates``.

    Only the driver error (``error.orig``) is inspected. ``str(error)`` is never
    matched because SQLAlchemy includes the SQL statement in it.
    """
    orig = getattr(error, "orig", None)
    if orig is None:
        return False
    code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    return code in sqlstates


def _columnar_column_list(columns: list[str]) -> str:
    """Build the ``columns`` argument of ``google_columnar_engine_add``.

    The server splits this string on ``,`` (and each entry on ``:``), strips
    whitespace and looks each name up verbatim with ``get_attnum``; it does
    not parse quoted identifiers. Names are therefore passed raw (as a bound
    parameter), and names the format cannot represent are rejected.
    """
    for column in columns:
        if not column or column != column.strip() or "," in column or ":" in column:
            raise ValueError(
                f"Column name {column!r} cannot be added to the columnar engine: "
                "names must be non-empty, without leading/trailing whitespace, "
                "and must not contain ',' or ':'."
            )
    return ",".join(columns)


def _require_name(value: Any, name: str) -> None:
    """Raise ValueError unless ``value`` is a non-empty, non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


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
        engine_args: Optional[Mapping[str, Any]] = None,
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
        engine_args: Optional[Mapping[str, Any]] = None,
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
            engine_args (Optional[Mapping[str, Any]]): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during its creation.

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
        engine_args: Optional[Mapping[str, Any]] = None,
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
            engine_args (Optional[Mapping[str, Any]]): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during its creation.

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

        engine_kwargs = dict(engine_args) if engine_args is not None else {}
        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
            **engine_kwargs,
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
        engine_args: Optional[Mapping[str, Any]] = None,
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
            engine_args (Optional[Mapping[str, Any]]): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during its creation.

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
        """Create an AlloyDBEngine instance from arguments.

        Args:
            url (str | URL): The URL used to connect to a database.

        Raises:
            ValueError: If not all database url arguments are specified.

        Returns:
            AlloyDBEngine: A newly created AlloyDBEngine instance.
        """
        return AlloyDBEngine.from_engine_args(url=url, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> AlloyDBEngine:
        """Create an AlloyDBEngine instance from arguments.

        Args:
            url (str | URL): The URL used to connect to a database.

        Raises:
            ValueError: If not all database url arguments are specified.

        Returns:
            AlloyDBEngine: A newly created AlloyDBEngine instance.
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
        """Create an AlloyDB table to store chat history.

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
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Creates a table for storing LangChain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (Optional[list[Column]]): A list of Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).
        """
        if metadata_columns is None:
            metadata_columns = []
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
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Creates a table for storing LangChain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (Optional[list[Column]]): A list of SQLAlchemy Columns
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
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Creates a table for storing LangChain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name to store the PgSQL database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (Optional[list[Column]]): A list of SQLAlchemy Columns
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
        """Create AlloyDB tables to store checkpoints.

        Args:
            table_name (str): The checkpoint table name. Default: "checkpoints".
            schema_name (str): The schema name to store checkpoint tables.
                Default: "public".

        Returns:
            None
        """
        self._run_as_sync(self._ainit_checkpoint_table(table_name, schema_name))

    async def _ais_function_missing(
        self,
        error: BaseException,
        sqlstates: set[str],
        schema: Optional[str],
        function: str,
    ) -> bool:
        """Return True if ``error`` means the extension function is not installed.

        The error must carry one of ``sqlstates`` and ``function`` (in
        ``schema``, or in any schema if None) must be absent from the catalog.
        The same SQLSTATEs are raised for other missing objects (for example a
        nonexistent schema passed by the caller), which must not be reported as
        a missing extension. If the catalog lookup itself fails, the SQLSTATE
        alone decides.
        """
        if not _is_missing_object_error(error, sqlstates):
            return False
        try:
            async with self._pool.connect() as conn:
                result = await conn.execute(
                    text(_FUNCTION_EXISTS_QUERY),
                    {"schema": schema, "name": function},
                )
                return not result.scalar()
        except Exception:
            return True

    async def _ainitialize_embeddings(
        self,
        table_name: str,
        model_id: str,
        content_column: str,
        embedding_column: str,
        schema_name: str = "public",
        overwrite: bool = False,
    ) -> None:
        """Call ``ai.initialize_embeddings`` for a table.

        See :meth:`ainitialize_embeddings` for the full description.
        """
        _require_name(table_name, "table_name")
        _require_name(model_id, "model_id")
        _require_name(content_column, "content_column")
        _require_name(embedding_column, "embedding_column")
        _require_name(schema_name, "schema_name")

        table_identifier = f"{_quote_ident(schema_name)}.{_quote_ident(table_name)}"
        if not overwrite:
            # ai.initialize_embeddings regenerates the embedding of every row,
            # including rows that already have one, so refuse to replace
            # existing vectors unless the caller asks for it.
            guard_query = (
                f"SELECT EXISTS (SELECT 1 FROM {table_identifier} "
                f"WHERE {_quote_ident(embedding_column)} IS NOT NULL)"
            )
            async with self._pool.connect() as conn:
                # Escape ":" so text() does not read a colon inside a quoted
                # identifier as a bind parameter.
                result = await conn.execute(text(guard_query.replace(":", r"\:")))
                has_embeddings = bool(result.scalar())
            if has_embeddings:
                raise ValueError(
                    f"Column '{embedding_column}' of table {table_identifier} "
                    "already contains embeddings. ai.initialize_embeddings "
                    "regenerates the embeddings of all rows, replacing them. "
                    "Pass overwrite=True to replace them, or use an empty "
                    "embedding column."
                )

        # ai.initialize_embeddings casts table_name to regclass, so pass the
        # quoted, schema-qualified identifier.
        query = "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
        try:
            async with self._pool.connect() as conn:
                # The procedure COMMITs internally (and then generates the
                # embeddings in batches), which PostgreSQL only allows when
                # CALL is not inside a transaction block. Run it in autocommit
                # mode so SQLAlchemy does not open one.
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                await conn.execute(
                    text(query),
                    {
                        "model_id": model_id,
                        "table_name": table_identifier,
                        "content_column": content_column,
                        "embedding_column": embedding_column,
                    },
                )
        except Exception as e:
            if await self._ais_function_missing(
                e,
                {_UNDEFINED_FUNCTION, _INVALID_SCHEMA_NAME},
                "ai",
                "initialize_embeddings",
            ):
                raise RuntimeError(_AI_INITIALIZE_EMBEDDINGS_MISSING_MSG) from e
            raise

    async def ainitialize_embeddings(
        self,
        table_name: str,
        model_id: str,
        content_column: str,
        embedding_column: str,
        schema_name: str = "public",
        overwrite: bool = False,
    ) -> None:
        """Generate and manage embeddings for a table with AlloyDB AI.

        Calls ``ai.initialize_embeddings``, which registers the table for
        automatic embedding management and then generates an embedding of
        ``content_column`` into ``embedding_column`` for **every** row, in
        batches, committing as it goes. Existing values in
        ``embedding_column`` are replaced, so by default this method refuses
        to run if the column already contains a non-NULL value (see
        ``overwrite``).

        ``embedding_column`` must already exist as a ``vector(N)`` column,
        where N is the model's output dimension (created for example with
        :meth:`ainit_vectorstore_table`); it is not created. If the table
        backs a vector store, use the same model as the vector store's
        embedding service: queries are embedded by that service, so
        embeddings from a different model make similarity search results
        meaningless. Use a separate column for a different model.

        The call runs on its own autocommit connection and cannot be part of
        a caller's transaction. It requires the
        ``google_ml_integration.enable_model_support`` and
        ``google_ml_integration.enable_faster_embedding_generation`` instance
        flags; the server's error is raised unchanged if they are off.

        The registration is committed before any embedding is generated and
        is not undone if the call fails later (for example because the model
        endpoint rejects the request). It adds a hidden
        ``google_ml_track_stale_embedding_<n>`` column, a trigger and an index
        to the table. Once the table is registered, calling this method again
        for the same table and embedding column fails with "already
        initialized". ``ai.drop_embedding_config`` removes the registration
        and these objects (the generated embeddings are kept);
        ``ai.refresh_embeddings`` completes an interrupted run.

        Args:
            table_name (str): The table to generate embeddings for.
            model_id (str): The ID of the embedding model, as registered in
                ``google_ml_integration`` (for example ``"text-embedding-005"``).
            content_column (str): The column containing the text to embed.
            embedding_column (str): The existing ``vector`` column to write
                the embeddings to.
            schema_name (str): The schema of the table. Default: "public".
            overwrite (bool): If False (the default), raise ValueError when
                ``embedding_column`` already contains a non-NULL value. If
                True, regenerate and replace all existing embeddings.

        Raises:
            ValueError: If an argument is empty, or if ``overwrite`` is False
                and ``embedding_column`` already contains embeddings.
            RuntimeError: If ``ai.initialize_embeddings`` is not available
                (``google_ml_integration`` missing or older than 1.5.6).
            sqlalchemy.exc.DBAPIError: Other database errors (for example a
                missing table or column, a non-vector embedding column, or an
                error from the model endpoint) are re-raised unchanged.
        """
        await self._run_as_async(
            self._ainitialize_embeddings(
                table_name,
                model_id,
                content_column,
                embedding_column,
                schema_name,
                overwrite,
            )
        )

    def initialize_embeddings(
        self,
        table_name: str,
        model_id: str,
        content_column: str,
        embedding_column: str,
        schema_name: str = "public",
        overwrite: bool = False,
    ) -> None:
        """Generate and manage embeddings for a table with AlloyDB AI.

        Calls ``ai.initialize_embeddings``, which generates an embedding for
        **every** row, replacing existing values in ``embedding_column``. By
        default this method refuses to run if the column already contains a
        non-NULL value (see ``overwrite``). ``embedding_column`` must already
        exist as a ``vector(N)`` column. If the table backs a vector store,
        use the same model as its embedding service. The call adds a hidden
        ``google_ml_track_stale_embedding_<n>`` column, a trigger and an index
        to the table (removed by ``ai.drop_embedding_config``). See
        :meth:`ainitialize_embeddings` for details.

        Args:
            table_name (str): The table to generate embeddings for.
            model_id (str): The ID of the embedding model, as registered in
                ``google_ml_integration`` (for example ``"text-embedding-005"``).
            content_column (str): The column containing the text to embed.
            embedding_column (str): The existing ``vector`` column to write
                the embeddings to.
            schema_name (str): The schema of the table. Default: "public".
            overwrite (bool): If False (the default), raise ValueError when
                ``embedding_column`` already contains a non-NULL value. If
                True, regenerate and replace all existing embeddings.

        Raises:
            ValueError: If an argument is empty, or if ``overwrite`` is False
                and ``embedding_column`` already contains embeddings.
            RuntimeError: If ``ai.initialize_embeddings`` is not available
                (``google_ml_integration`` missing or older than 1.5.6).
            sqlalchemy.exc.DBAPIError: Other database errors are re-raised
                unchanged.
        """
        self._run_as_sync(
            self._ainitialize_embeddings(
                table_name,
                model_id,
                content_column,
                embedding_column,
                schema_name,
                overwrite,
            )
        )

    async def _aenable_columnar_engine(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        schema_name: str = "public",
    ) -> None:
        """Call ``google_columnar_engine_add`` for a table.

        See :meth:`aenable_columnar_engine` for the full description.
        """
        _require_name(table_name, "table_name")
        _require_name(schema_name, "schema_name")
        table_identifier = f"{_quote_ident(schema_name)}.{_quote_ident(table_name)}"

        if columns is not None:
            # A bare str would otherwise be split into single characters.
            col_names = [] if isinstance(columns, str) else list(columns)
            if isinstance(columns, str) or not all(
                isinstance(column, str) for column in col_names
            ):
                raise ValueError(
                    "columns must be a list of column names (strings), "
                    f"not {columns!r}."
                )
            if not col_names:
                raise ValueError(
                    "columns must not be empty; pass None to add every "
                    "non-vector column."
                )
        else:
            async with self._pool.connect() as conn:
                col_result = await conn.execute(
                    text(_DEFAULT_COLUMNAR_COLUMNS_QUERY),
                    {
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "tracking_column_pattern": _TRACKING_COLUMN_PATTERN,
                    },
                )
                col_names = [row[0] for row in col_result.fetchall()]
            if not col_names:
                raise ValueError(
                    f"No non-vector columns were found for table "
                    f"{table_identifier}; the table may not exist, may not be "
                    "visible to the current user, or may only have vector "
                    "columns."
                )

        query = "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
        params = {
            "table_name": table_identifier,
            "columns": _columnar_column_list(col_names),
        }
        try:
            async with self._pool.connect() as conn:
                result = await conn.execute(text(query), params)
                size_mb = result.scalar()
                await conn.commit()
        except Exception as e:
            if await self._ais_function_missing(
                e, {_UNDEFINED_FUNCTION}, None, "google_columnar_engine_add"
            ):
                raise RuntimeError(_COLUMNAR_ENGINE_MISSING_MSG) from e
            raise
        if not size_mb:
            logger.warning(
                "google_columnar_engine_add returned %r for %s (columns: %s). "
                "The relation may not have been added to the columnar engine; "
                "check the server notices / log for the reason (for example "
                "the columnar engine is disabled or a lock was not available). "
                "Very small tables can also legitimately report 0 MB.",
                size_mb,
                table_identifier,
                params["columns"],
            )

    async def aenable_columnar_engine(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        schema_name: str = "public",
    ) -> None:
        """Add a table's columns to the AlloyDB columnar engine.

        Calls ``google_columnar_engine_add``. This is a one-time action on
        the node the engine is connected to: the columns are not added on
        other nodes (for example read pool instances), and they are not kept
        across instance restarts. To keep columns in the columnar engine
        durably, list them in the ``google_columnar_engine.relations``
        instance flag instead.

        The server reports most failures (columnar engine disabled, lock not
        available, ...) as a WARNING and returns 0 instead of raising; when it
        returns 0 this method logs a warning. A 0 result can also mean the
        populated data rounds down to 0 MB (very small tables), so it is not
        treated as an error.

        Args:
            table_name (str): The table to add to the columnar engine.
            columns (Optional[list[str]]): The columns to add. If None, every
                column of the table is added except pgvector-typed columns
                (``vector``, ``halfvec``, ``sparsevec``) and the
                ``google_ml_track_stale_embedding_<n>`` columns added by
                :meth:`ainitialize_embeddings`. To include a vector column,
                list it explicitly.
            schema_name (str): The schema of the table. Default: "public".

        Raises:
            ValueError: If ``table_name`` or ``schema_name`` is empty, if
                ``columns`` is not a list of strings or is an empty list, if a
                column name contains ','
                or ':' or has leading/trailing whitespace (the columnar
                engine's column list format cannot express such names), or if
                ``columns`` is None and the table has no eligible column (or
                is not visible).
            RuntimeError: If the columnar engine is not available on the instance.
        """
        await self._run_as_async(
            self._aenable_columnar_engine(table_name, columns, schema_name)
        )

    def enable_columnar_engine(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        schema_name: str = "public",
    ) -> None:
        """Add a table's columns to the AlloyDB columnar engine.

        Calls ``google_columnar_engine_add``. This is a one-time action on
        the connected node; the columns are not kept across instance
        restarts (use the ``google_columnar_engine.relations`` instance flag
        for that). Logs a warning if the server returns 0. See
        :meth:`aenable_columnar_engine` for details.

        Args:
            table_name (str): The table to add to the columnar engine.
            columns (Optional[list[str]]): The columns to add. If None, every
                column except pgvector-typed columns and the
                ``google_ml_track_stale_embedding_<n>`` columns is added.
            schema_name (str): The schema of the table. Default: "public".

        Raises:
            ValueError: If an argument is empty or invalid, or if ``columns``
                is None and the table has no eligible column.
            RuntimeError: If the columnar engine is not available on the instance.
        """
        self._run_as_sync(
            self._aenable_columnar_engine(table_name, columns, schema_name)
        )

    async def _arun_auto_columnarization(self) -> list[dict]:
        """Call ``google_columnar_engine_recommend('AUTO_COLUMNARIZATION')``.

        See :meth:`arun_auto_columnarization` for the full description.
        """
        query = "SELECT * FROM google_columnar_engine_recommend('AUTO_COLUMNARIZATION')"
        try:
            async with self._pool.connect() as conn:
                result = await conn.execute(text(query))
                rows = [dict(row) for row in result.mappings()]
                await conn.commit()
                return rows
        except Exception as e:
            if await self._ais_function_missing(
                e, {_UNDEFINED_FUNCTION}, None, "google_columnar_engine_recommend"
            ):
                raise RuntimeError(_COLUMNAR_ENGINE_MISSING_MSG) from e
            raise

    async def arun_auto_columnarization(self) -> list[dict]:
        """Run the columnar engine's auto-columnarization once.

        Calls
        ``google_columnar_engine_recommend(mode => 'AUTO_COLUMNARIZATION')``,
        which analyzes the workload of the whole instance and populates the
        columnar engine with the columns it recommends, for any table. This
        is instance-wide and runs once; it does not create a recurring
        schedule (that is the instance's own auto-columnarization setting).

        Returns:
            list[dict]: The rows returned by ``google_columnar_engine_recommend``,
            each with ``total_size_in_mb`` (INT8) and ``columns`` (TEXT, the
            recommended columns).

        Raises:
            RuntimeError: If the columnar engine is not available on the instance.
        """
        return await self._run_as_async(self._arun_auto_columnarization())

    def run_auto_columnarization(self) -> list[dict]:
        """Run the columnar engine's auto-columnarization once.

        Instance-wide and one-time. See :meth:`arun_auto_columnarization`.

        Returns:
            list[dict]: The rows returned by ``google_columnar_engine_recommend``,
            each with ``total_size_in_mb`` and ``columns``.

        Raises:
            RuntimeError: If the columnar engine is not available on the instance.
        """
        return self._run_as_sync(self._arun_auto_columnarization())

    async def _adefine_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        embedding_model: Optional[str] = None,
    ) -> list[dict]:
        """Call ``vector_assist.define_spec``.

        See :meth:`adefine_vector_assist_spec` for the full description.
        """
        _require_name(table_name, "table_name")
        _require_name(embedding_column, "embedding_column")
        _require_name(schema_name, "schema_name")
        query = (
            "SELECT * FROM vector_assist.define_spec("
            "table_name => :table_name, schema_name => :schema_name, "
            "vector_column_name => :embedding_column"
        )
        params = {
            "table_name": table_name,
            "schema_name": schema_name,
            "embedding_column": embedding_column,
        }
        if embedding_model is not None:
            query += ", embedding_model => :embedding_model"
            params["embedding_model"] = embedding_model
        query += ")"
        try:
            async with self._pool.connect() as conn:
                result = await conn.execute(text(query), params)
                rows = [dict(row) for row in result.mappings()]
                await conn.commit()
                return rows
        except Exception as e:
            if await self._ais_function_missing(
                e,
                {_UNDEFINED_FUNCTION, _INVALID_SCHEMA_NAME},
                "vector_assist",
                "define_spec",
            ):
                raise RuntimeError(_VECTOR_ASSIST_MISSING_MSG) from e
            raise

    async def adefine_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        embedding_model: Optional[str] = None,
    ) -> list[dict]:
        """Define a Vector Assist spec for a table's vector column.

        Args:
            table_name (str): The table containing the vector column.
            embedding_column (str): The vector column.
            schema_name (str): The schema of the table. Default: "public".
            embedding_model (Optional[str]): The model ID (as registered in
                ``google_ml_integration``, e.g. ``"text-embedding-005"``) that
                was used to generate the embeddings in the table. Vector Assist
                requires it when the table already contains embeddings.
                Defaults to None.

        Returns:
            list[dict]: The recommendation rows returned by
            ``vector_assist.define_spec`` (columns of
            ``vector_assist.recommendations``, e.g. ``recommendation_id``,
            ``vector_spec_id``, ``query``).

        Raises:
            ValueError: If ``table_name``, ``embedding_column`` or
                ``schema_name`` is empty.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return await self._run_as_async(
            self._adefine_vector_assist_spec(
                table_name, embedding_column, schema_name, embedding_model
            )
        )

    def define_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        embedding_model: Optional[str] = None,
    ) -> list[dict]:
        """Define a Vector Assist spec for a table's vector column.

        Args:
            table_name (str): The table containing the vector column.
            embedding_column (str): The vector column.
            schema_name (str): The schema of the table. Default: "public".
            embedding_model (Optional[str]): The model ID that was used to
                generate the embeddings in the table. Vector Assist requires
                it when the table already contains embeddings. Defaults to None.

        Returns:
            list[dict]: The recommendation rows returned by
            ``vector_assist.define_spec``.

        Raises:
            ValueError: If ``table_name``, ``embedding_column`` or
                ``schema_name`` is empty.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return self._run_as_sync(
            self._adefine_vector_assist_spec(
                table_name, embedding_column, schema_name, embedding_model
            )
        )

    async def _afetch_latest_vector_assist_spec_id(
        self,
        conn: AsyncConnection,
        table_name: str,
        embedding_column: str,
        schema_name: str,
    ) -> Optional[str]:
        """Return the id of the most recently defined spec, or None."""
        result = await conn.execute(
            text(_LATEST_VECTOR_ASSIST_SPEC_QUERY),
            {
                "table_name": table_name,
                "schema_name": schema_name,
                "embedding_column": embedding_column,
            },
        )
        row = result.mappings().first()
        return row["spec_id"] if row else None

    async def _aapply_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        spec_id: Optional[str] = None,
    ) -> Optional[bool]:
        """Call ``vector_assist.apply_spec``.

        See :meth:`aapply_vector_assist_spec` for the full description.
        """
        _require_name(table_name, "table_name")
        _require_name(embedding_column, "embedding_column")
        _require_name(schema_name, "schema_name")
        try:
            async with self._pool.connect() as conn:
                if spec_id is None:
                    spec_id = await self._afetch_latest_vector_assist_spec_id(
                        conn, table_name, embedding_column, schema_name
                    )
                    if spec_id is None:
                        raise ValueError(
                            "No Vector Assist spec found for table "
                            f"'{schema_name}.{table_name}' and column "
                            f"'{embedding_column}'. Call "
                            "define_vector_assist_spec() first to create a spec."
                        )
                result = await conn.execute(
                    text("SELECT vector_assist.apply_spec(spec_id => :spec_id)"),
                    {"spec_id": spec_id},
                )
                applied = result.scalar()
                await conn.commit()
                return applied
        except Exception as e:
            if await self._ais_function_missing(
                e,
                {_UNDEFINED_FUNCTION, _INVALID_SCHEMA_NAME, _UNDEFINED_TABLE},
                "vector_assist",
                "apply_spec",
            ):
                raise RuntimeError(_VECTOR_ASSIST_MISSING_MSG) from e
            raise

    async def aapply_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        spec_id: Optional[str] = None,
    ) -> Optional[bool]:
        """Apply a Vector Assist spec.

        Applying a spec executes every recommendation of the spec. Depending on
        the spec these can include database-wide statements (for example
        ``ALTER EXTENSION vector UPDATE``) in addition to table-level ones such
        as ``CREATE INDEX``. Review them with
        :meth:`aget_vector_assist_recommendations` first.

        Args:
            table_name (str): The table the spec was defined for.
            embedding_column (str): The vector column the spec was defined for.
            schema_name (str): The schema of the table. Default: "public".
            spec_id (Optional[str]): The spec to apply. If None, the most
                recently defined spec for ``schema_name.table_name`` and
                ``embedding_column`` is applied (the same spec
                :meth:`aget_vector_assist_recommendations` returns
                recommendations for).

        Returns:
            Optional[bool]: The value returned by ``vector_assist.apply_spec``:
            True if all recommendations were applied, False otherwise, or None
            if the spec has no recommendations.

        Raises:
            ValueError: If ``table_name``, ``embedding_column`` or
                ``schema_name`` is empty, or if ``spec_id`` is None and no
                spec has been defined for the table and column.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return await self._run_as_async(
            self._aapply_vector_assist_spec(
                table_name, embedding_column, schema_name, spec_id
            )
        )

    def apply_vector_assist_spec(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
        spec_id: Optional[str] = None,
    ) -> Optional[bool]:
        """Apply a Vector Assist spec.

        Applying a spec executes every recommendation of the spec, which can
        include database-wide statements. Review them with
        :meth:`get_vector_assist_recommendations` first.

        Args:
            table_name (str): The table the spec was defined for.
            embedding_column (str): The vector column the spec was defined for.
            schema_name (str): The schema of the table. Default: "public".
            spec_id (Optional[str]): The spec to apply. If None, the most
                recently defined spec for the table and column is applied.

        Returns:
            Optional[bool]: The value returned by ``vector_assist.apply_spec``.

        Raises:
            ValueError: If an argument is empty, or if ``spec_id`` is None and
                no spec has been defined for the table and column.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return self._run_as_sync(
            self._aapply_vector_assist_spec(
                table_name, embedding_column, schema_name, spec_id
            )
        )

    async def _aget_vector_assist_recommendations(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
    ) -> list[dict]:
        """Call ``vector_assist.get_recommendations`` for the latest spec.

        See :meth:`aget_vector_assist_recommendations` for the full description.
        """
        _require_name(table_name, "table_name")
        _require_name(embedding_column, "embedding_column")
        _require_name(schema_name, "schema_name")
        try:
            async with self._pool.connect() as conn:
                spec_id = await self._afetch_latest_vector_assist_spec_id(
                    conn, table_name, embedding_column, schema_name
                )
                if spec_id is None:
                    logger.warning(
                        "No Vector Assist spec found for table '%s.%s' and "
                        "column '%s'. Call define_vector_assist_spec() first "
                        "to create a spec.",
                        schema_name,
                        table_name,
                        embedding_column,
                    )
                    return []

                query = "SELECT * FROM vector_assist.get_recommendations(spec_id => :spec_id)"
                result = await conn.execute(text(query), {"spec_id": spec_id})
                return [dict(row) for row in result.mappings()]
        except Exception as e:
            if await self._ais_function_missing(
                e,
                {_UNDEFINED_FUNCTION, _INVALID_SCHEMA_NAME, _UNDEFINED_TABLE},
                "vector_assist",
                "get_recommendations",
            ):
                raise RuntimeError(_VECTOR_ASSIST_MISSING_MSG) from e
            raise

    async def aget_vector_assist_recommendations(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
    ) -> list[dict]:
        """Get the recommendations of the latest Vector Assist spec.

        Uses the most recently defined spec for ``schema_name.table_name``
        and ``embedding_column``. No new spec is defined.

        Args:
            table_name (str): The table the spec was defined for.
            embedding_column (str): The vector column the spec was defined for.
            schema_name (str): The schema of the table. Default: "public".

        Returns:
            list[dict]: Rows of ``vector_assist.get_recommendations``, or an
            empty list (and a logged warning) if no spec has been defined.

        Raises:
            ValueError: If ``table_name``, ``embedding_column`` or
                ``schema_name`` is empty.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return await self._run_as_async(
            self._aget_vector_assist_recommendations(
                table_name, embedding_column, schema_name
            )
        )

    def get_vector_assist_recommendations(
        self,
        table_name: str,
        embedding_column: str,
        schema_name: str = "public",
    ) -> list[dict]:
        """Get the recommendations of the latest Vector Assist spec.

        Args:
            table_name (str): The table the spec was defined for.
            embedding_column (str): The vector column the spec was defined for.
            schema_name (str): The schema of the table. Default: "public".

        Returns:
            list[dict]: Rows of ``vector_assist.get_recommendations``, or an
            empty list if no spec has been defined.

        Raises:
            ValueError: If ``table_name``, ``embedding_column`` or
                ``schema_name`` is empty.
            RuntimeError: If the Vector Assist extension is not installed.
        """
        return self._run_as_sync(
            self._aget_vector_assist_recommendations(
                table_name, embedding_column, schema_name
            )
        )

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
