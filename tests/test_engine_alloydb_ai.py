# Copyright 2026 Google LLC
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

"""Tests for the AlloyDB AI operations of AlloyDBEngine: ai.initialize_embeddings,
the columnar engine, auto-columnarization and Vector Assist."""

import asyncio
import inspect
import logging
import os
import uuid
from typing import Any, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.exc import ProgrammingError

from langchain_google_alloydb_pg import AlloyDBEngine

VECTOR_SIZE = 768  # Output dimension of text-embedding-005.
EMBEDDING_MODEL_ID = "text-embedding-005"
ENGINE_LOGGER = "langchain_google_alloydb_pg.engine"

# SQLSTATEs raised when an extension function/schema/table does not exist.
MISSING_OBJECT_SQLSTATES = {"42883", "3F000", "42P01"}


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


# asyncpg connections are bound to the event loop that opened them, so every
# helper that uses engine._pool runs on the engine's background loop.
async def aexecute(engine: AlloyDBEngine, query: str, params: Any = None) -> None:
    async def run() -> None:
        async with engine._pool.connect() as conn:
            await conn.execute(text(query), params or {})
            await conn.commit()

    await engine._run_as_async(run())


async def afetch(
    engine: AlloyDBEngine, query: str, params: Any = None
) -> Sequence[RowMapping]:
    async def run() -> Sequence[RowMapping]:
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query), params or {})
            return result.mappings().fetchall()

    return await engine._run_as_async(run())


def _db_error_orig(error: BaseException) -> Any:
    """Return the DB-API error behind ``error`` (unwrapping the library's RuntimeError)."""
    if isinstance(error, RuntimeError) and error.__cause__ is not None:
        error = error.__cause__
    return getattr(error, "orig", None)


def _sqlstate(orig: Any) -> Optional[str]:
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


def _skip_if_columnar_engine_unavailable(error: BaseException) -> None:
    orig = _db_error_orig(error)
    if orig is None:
        return
    code = _sqlstate(orig)
    # 55000 object_not_in_prerequisite_state: google_columnar_engine.enabled is off.
    if code in MISSING_OBJECT_SQLSTATES or (
        code == "55000" and "google_columnar_engine" in str(orig)
    ):
        pytest.skip(f"Columnar engine not available on instance: {orig}")


def _skip_if_vector_assist_unavailable(error: BaseException) -> None:
    orig = _db_error_orig(error)
    if orig is None:
        return
    if _sqlstate(orig) in MISSING_OBJECT_SQLSTATES or "vector_assist.enabled" in str(
        orig
    ):
        pytest.skip(f"Vector Assist not available on instance: {orig}")


async def _acolumnar_columns(
    engine: AlloyDBEngine, table_name: str, schema_name: str = "public"
) -> set[str]:
    """Names of the columns of ``table_name`` listed in ``g_columnar_columns``."""
    rows = await afetch(
        engine,
        "SELECT column_name FROM g_columnar_columns "
        "WHERE schema_name = :s AND relation_name = :t",
        {"s": schema_name, "t": table_name},
    )
    return {row["column_name"] for row in rows}


async def _adrop_from_columnar_engine(engine: AlloyDBEngine, table_name: str) -> None:
    """Best-effort removal of ``public.table_name`` from the columnar engine."""
    try:
        await aexecute(
            engine,
            "SELECT google_columnar_engine_drop(relation => :r)",
            {"r": f'"public"."{table_name}"'},
        )
    except Exception:
        pass


async def _amodel_registered(engine: AlloyDBEngine, model_id: str) -> bool:
    """Whether ``model_id`` is registered in google_ml_integration."""
    try:
        rows = await afetch(
            engine,
            "SELECT 1 FROM google_ml.model_info_view WHERE model_id = :m",
            {"m": model_id},
        )
        return len(rows) > 0
    except Exception:
        return False


async def _adrop_embedding_config(
    engine: AlloyDBEngine, table_name: str, embedding_column: str
) -> None:
    """Best-effort ai.drop_embedding_config (it must run outside a transaction)."""

    async def run() -> None:
        async with engine._pool.connect() as conn:
            await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(
                text("CALL ai.drop_embedding_config(:t, :c)"),
                {"t": f'"public"."{table_name}"', "c": embedding_column},
            )

    try:
        await engine._run_as_async(run())
    except Exception:
        pass


async def _atable_columns(engine: AlloyDBEngine, table_name: str) -> list[str]:
    rows = await afetch(
        engine,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :t "
        "ORDER BY ordinal_position",
        {"t": table_name},
    )
    return [row["column_name"] for row in rows]


async def _arestrict_vector_assist_spec_to_table(
    engine: AlloyDBEngine, spec_id: str, table_name: str
) -> list[str]:
    """Neutralize every recommendation of ``spec_id`` that is not a CREATE INDEX
    on ``table_name``, so applying the spec cannot change the shared instance.

    Vector Assist can recommend database-wide statements (CREATE / ALTER
    EXTENSION ... UPDATE for vector and google_ml_integration) and session
    settings that would persist on the pooled connection. Those are replaced
    with ``SELECT 1`` via the public ``vector_assist.modify_recommendation``.
    Returns the ids of the recommendations that were kept.
    """

    async def run() -> list[str]:
        kept = []
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT recommendation_id, query "
                    "FROM vector_assist.get_recommendations(spec_id => :spec_id)"
                ),
                {"spec_id": spec_id},
            )
            for rec in result.mappings().all():
                query = rec["query"] or ""
                if query.lstrip().upper().startswith("CREATE INDEX") and (
                    table_name in query
                ):
                    kept.append(rec["recommendation_id"])
                    continue
                modified = await conn.execute(
                    text(
                        "SELECT vector_assist.modify_recommendation("
                        "recommendation_id => :rid, modified_query => 'SELECT 1')"
                    ),
                    {"rid": rec["recommendation_id"]},
                )
                assert modified.scalar() is True
            await conn.commit()
        return kept

    return await engine._run_as_async(run())


async def _adelete_vector_assist_specs(engine: AlloyDBEngine, table_name: str) -> None:
    """Delete every Vector Assist spec (and its recommendations) of ``table_name``."""

    async def run() -> None:
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT spec_id FROM vector_assist.vector_specs WHERE table_name = :t"
                ),
                {"t": table_name},
            )
            spec_ids: list[str] = list(result.scalars().all())
            for spec_id in spec_ids:
                await conn.execute(
                    text(
                        "SELECT * FROM vector_assist.delete_spec(spec_id => :spec_id)"
                    ),
                    {"spec_id": spec_id},
                )
            await conn.commit()

    await engine._run_as_async(run())


def _new_table_name(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("PROJECT_ID"),
    reason="Requires live AlloyDB instance (PROJECT_ID not set)",
)
class TestEngineAlloyDBAIIntegration:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    async def _acreate_table(
        self, engine: AlloyDBEngine, table_name: str, rows: int, embed: bool
    ) -> None:
        """Create a table with two vector columns and ``rows`` rows.

        ``embedding`` is filled with random vectors if ``embed`` is True, and
        left NULL otherwise.
        """
        await aexecute(
            engine,
            f'CREATE TABLE "{table_name}" (id uuid PRIMARY KEY, content text, '
            f"embedding vector({VECTOR_SIZE}), other_vec vector(3), meta jsonb)",
        )
        embedding_expr = (
            f"(SELECT array_agg((random() * 2 - 1)::float4)::vector({VECTOR_SIZE}) "
            f"FROM generate_series(1, {VECTOR_SIZE}) WHERE i IS NOT NULL)"
            if embed
            else "NULL"
        )
        await aexecute(
            engine,
            f'INSERT INTO "{table_name}" (id, content, embedding, other_vec, meta) '
            f"SELECT gen_random_uuid(), 'Content ' || i, {embedding_expr}, "
            "'[1,2,3]'::vector, '{\"page\": 1}'::jsonb "
            f"FROM generate_series(1, {rows}) AS i",
        )

    async def test_live_enable_columnar_engine(self, engine):
        """Explicit and default column lists land in g_columnar_columns; the
        default excludes every vector column."""
        table_name = _new_table_name("ce_live_")
        await self._acreate_table(engine, table_name, rows=200, embed=True)
        try:
            try:
                await engine.aenable_columnar_engine(table_name, ["content"])
            except Exception as e:
                _skip_if_columnar_engine_unavailable(e)
                raise
            assert await _acolumnar_columns(engine, table_name) == {"content"}

            await engine.aenable_columnar_engine(table_name)
            assert await _acolumnar_columns(engine, table_name) == {
                "id",
                "content",
                "meta",
            }
        finally:
            await _adrop_from_columnar_engine(engine, table_name)
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_enable_columnar_engine_sync(self, engine):
        table_name = _new_table_name("ce_live_sync_")
        await self._acreate_table(engine, table_name, rows=200, embed=False)
        try:
            try:
                await asyncio.to_thread(engine.enable_columnar_engine, table_name)
            except Exception as e:
                _skip_if_columnar_engine_unavailable(e)
                raise
            assert await _acolumnar_columns(engine, table_name) == {
                "id",
                "content",
                "meta",
            }
        finally:
            await _adrop_from_columnar_engine(engine, table_name)
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_initialize_then_default_columnar_excludes_tracking_column(
        self, engine
    ):
        """Regression test: after ai.initialize_embeddings adds its hidden
        google_ml_track_stale_embedding_<n> column, the default columnar column
        list still excludes it and both vector columns."""
        if not await _amodel_registered(engine, EMBEDDING_MODEL_ID):
            pytest.skip(f"Model {EMBEDDING_MODEL_ID} is not registered")
        table_name = _new_table_name("ai_ce_live_")
        await self._acreate_table(engine, table_name, rows=5, embed=False)
        try:
            try:
                await engine.ainitialize_embeddings(
                    table_name, EMBEDDING_MODEL_ID, "content", "embedding"
                )
            except RuntimeError as e:
                pytest.skip(f"ai.initialize_embeddings not available: {e}")
            rows = await afetch(
                engine,
                f'SELECT count(*) AS total, count(embedding) AS embedded FROM "{table_name}"',
            )
            assert rows[0]["total"] == rows[0]["embedded"] == 5

            columns = await _atable_columns(engine, table_name)
            tracking = [
                c for c in columns if c.startswith("google_ml_track_stale_embedding_")
            ]
            assert len(tracking) == 1, columns

            try:
                await engine.aenable_columnar_engine(table_name)
            except Exception as e:
                _skip_if_columnar_engine_unavailable(e)
                raise
            assert await _acolumnar_columns(engine, table_name) == {
                "id",
                "content",
                "meta",
            }
        finally:
            await _adrop_from_columnar_engine(engine, table_name)
            await _adrop_embedding_config(engine, table_name, "embedding")
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_initialize_embeddings_overwrite_guard(self, engine):
        """With pre-filled vectors, overwrite=False refuses without touching the
        table; overwrite=True regenerates every row, replacing the old vector."""
        if not await _amodel_registered(engine, EMBEDDING_MODEL_ID):
            pytest.skip(f"Model {EMBEDDING_MODEL_ID} is not registered")
        table_name = _new_table_name("ai_guard_live_")
        await self._acreate_table(engine, table_name, rows=3, embed=False)
        try:
            # Pre-fill a single row with a known vector.
            await aexecute(
                engine,
                f'UPDATE "{table_name}" SET embedding = '
                f"array_fill(0.5::float4, ARRAY[{VECTOR_SIZE}])::vector({VECTOR_SIZE}) "
                f'WHERE id = (SELECT min(id::text)::uuid FROM "{table_name}")',
            )
            columns_before = await _atable_columns(engine, table_name)

            with pytest.raises(ValueError, match="already contains embeddings"):
                await engine.ainitialize_embeddings(
                    table_name, EMBEDDING_MODEL_ID, "content", "embedding"
                )
            with pytest.raises(ValueError, match="already contains embeddings"):
                await asyncio.to_thread(
                    engine.initialize_embeddings,
                    table_name,
                    EMBEDDING_MODEL_ID,
                    "content",
                    "embedding",
                )
            # Nothing was registered: no tracking column, same vectors.
            assert await _atable_columns(engine, table_name) == columns_before
            rows = await afetch(
                engine,
                f'SELECT count(embedding) AS embedded FROM "{table_name}"',
            )
            assert rows[0]["embedded"] == 1

            try:
                await engine.ainitialize_embeddings(
                    table_name,
                    EMBEDDING_MODEL_ID,
                    "content",
                    "embedding",
                    overwrite=True,
                )
            except RuntimeError as e:
                pytest.skip(f"ai.initialize_embeddings not available: {e}")
            rows = await afetch(
                engine,
                "SELECT count(*) AS total, count(embedding) AS embedded, "
                f"count(*) FILTER (WHERE embedding = array_fill(0.5::float4, "
                f"ARRAY[{VECTOR_SIZE}])::vector({VECTOR_SIZE})) AS unchanged "
                f'FROM "{table_name}"',
            )
            assert rows[0]["total"] == rows[0]["embedded"] == 3
            assert rows[0]["unchanged"] == 0
        finally:
            await _adrop_embedding_config(engine, table_name, "embedding")
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_initialize_embeddings_sync(self, engine):
        """The sync wrapper runs the CALL outside a transaction block too."""
        if not await _amodel_registered(engine, EMBEDDING_MODEL_ID):
            pytest.skip(f"Model {EMBEDDING_MODEL_ID} is not registered")
        table_name = _new_table_name("ai_sync_live_")
        await self._acreate_table(engine, table_name, rows=3, embed=False)
        try:
            try:
                await asyncio.to_thread(
                    engine.initialize_embeddings,
                    table_name,
                    EMBEDDING_MODEL_ID,
                    "content",
                    "embedding",
                )
            except RuntimeError as e:
                pytest.skip(f"ai.initialize_embeddings not available: {e}")
            rows = await afetch(
                engine,
                f'SELECT count(*) AS total, count(embedding) AS embedded FROM "{table_name}"',
            )
            assert rows[0]["total"] == rows[0]["embedded"] == 3
        finally:
            await _adrop_embedding_config(engine, table_name, "embedding")
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_initialize_embeddings_missing_column_is_not_masked(
        self, engine
    ):
        """A missing embedding column surfaces as the server's error, not as a
        'not available' RuntimeError, with and without the overwrite guard."""
        table_name = _new_table_name("ai_missing_live_")
        await aexecute(engine, f'CREATE TABLE "{table_name}" (id int, content text)')
        try:
            for overwrite in (False, True):
                with pytest.raises(sqlalchemy.exc.DBAPIError):
                    await engine.ainitialize_embeddings(
                        table_name,
                        EMBEDDING_MODEL_ID,
                        "content",
                        "no_such_column",
                        overwrite=overwrite,
                    )
        finally:
            await _adrop_embedding_config(engine, table_name, "no_such_column")
            await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')

    async def test_live_run_auto_columnarization(self, engine):
        """Auto-columnarization returns recommendation rows.

        google_columnar_engine_recommend is instance-wide; it may populate the
        columnar engine with columns of other tables, which is the function's
        purpose and not undoable per table.
        """
        try:
            rows = await engine.arun_auto_columnarization()
        except Exception as e:
            _skip_if_columnar_engine_unavailable(e)
            raise
        assert isinstance(rows, list)
        for row in rows:
            assert set(row) == {"total_size_in_mb", "columns"}

        rows = await asyncio.to_thread(engine.run_auto_columnarization)
        assert isinstance(rows, list)

    async def test_live_vector_assist(self, engine):
        """Define, inspect and apply a Vector Assist spec.

        Only recommendations scoped to this test's own table (CREATE INDEX) are
        executed by apply; the others are replaced with no-ops first.
        Afterwards the spec is deleted with vector_assist.delete_spec and the
        table is dropped. delete_spec only soft-deletes the extension's
        internal copy (vector_assist.vector_specs_internal.deleted_at); there
        is no public API to remove that row.
        """
        table_name = _new_table_name("va_live_")
        spec_defined = False
        await self._acreate_table(engine, table_name, rows=100, embed=True)
        try:
            # No spec yet: empty list.
            try:
                assert (
                    await engine.aget_vector_assist_recommendations(
                        table_name, "embedding"
                    )
                    == []
                )
                specs = await engine.adefine_vector_assist_spec(
                    table_name, "embedding", embedding_model=EMBEDDING_MODEL_ID
                )
            except Exception as e:
                _skip_if_vector_assist_unavailable(e)
                raise
            spec_defined = True
            assert len(specs) > 0
            spec_ids = {row["vector_spec_id"] for row in specs}
            assert len(spec_ids) == 1
            assert all(row["recommendation_id"] for row in specs)
            (spec_id,) = spec_ids

            recs = await engine.aget_vector_assist_recommendations(
                table_name, "embedding"
            )
            assert {row["vector_spec_id"] for row in recs} == spec_ids
            assert {row["recommendation_id"] for row in recs} == {
                row["recommendation_id"] for row in specs
            }
            sync_recs = await asyncio.to_thread(
                engine.get_vector_assist_recommendations, table_name, "embedding"
            )
            assert {row["recommendation_id"] for row in sync_recs} == {
                row["recommendation_id"] for row in recs
            }

            kept = await _arestrict_vector_assist_spec_to_table(
                engine, spec_id, table_name
            )
            # No spec_id: resolves the same (latest) spec as get_recommendations.
            assert await engine.aapply_vector_assist_spec(table_name, "embedding")
            recs = await engine.aget_vector_assist_recommendations(
                table_name, "embedding"
            )
            assert {row["vector_spec_id"] for row in recs} == spec_ids
            assert all(row["applied"] for row in recs)
            if kept:
                rows = await afetch(
                    engine,
                    "SELECT count(*) AS n FROM pg_index "
                    f'WHERE indrelid = \'"public"."{table_name}"\'::regclass '
                    "AND NOT indisprimary",
                )
                assert rows[0]["n"] >= 1
        finally:
            try:
                if spec_defined:
                    await _adelete_vector_assist_specs(engine, table_name)
            finally:
                await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}" CASCADE')


class _FakeDBAPIError(Exception):
    """Stand-in for a DB-API error carrying a SQLSTATE (like SQLAlchemy's asyncpg adapter)."""

    def __init__(
        self, message: str, sqlstate: Optional[str], pgcode: Optional[str] = None
    ):
        super().__init__(message)
        if sqlstate is not None:
            self.sqlstate = sqlstate
        self.pgcode = pgcode if pgcode is not None else sqlstate


INITIALIZE_SQL = "CALL ai.initialize_embeddings(:model_id, :table_name, :content_column, :embedding_column)"
COLUMNAR_ADD_SQL = (
    "SELECT google_columnar_engine_add(relation => :table_name, columns => :columns)"
)
RECOMMEND_SQL = "SELECT * FROM google_columnar_engine_recommend('AUTO_COLUMNARIZATION')"
DEFINE_SPEC_SQL = (
    "SELECT * FROM vector_assist.define_spec(table_name => :table_name, "
    "schema_name => :schema_name, vector_column_name => :embedding_column)"
)
LATEST_SPEC_SQL = (
    "SELECT spec_id FROM vector_assist.vector_specs "
    "WHERE table_name = :table_name "
    "AND schema_name = COALESCE(CAST(:schema_name AS TEXT), current_schema()) "
    "AND vector_column_name = :embedding_column "
    "ORDER BY created_at DESC, spec_id DESC LIMIT 1"
)
DEFAULT_COLUMNS_SQL = (
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_schema = :schema_name AND table_name = :table_name "
    "AND udt_name NOT IN ('vector', 'halfvec', 'sparsevec') "
    "AND column_name NOT LIKE :tracking_column_pattern "
    "ORDER BY ordinal_position"
)
TRACKING_PATTERN = r"google\_ml\_track\_stale\_embedding\_%"

# The first SQL statement each method sends (with overwrite=True for
# initialize, so the guard query is skipped); every statement names the
# extension object.
_EXTENSION_CALLS = {
    "ainitialize_embeddings": (
        {
            "table_name": "t",
            "model_id": "m",
            "content_column": "content",
            "embedding_column": "embedding",
            "overwrite": True,
        },
        INITIALIZE_SQL,
    ),
    "aenable_columnar_engine": (
        {"table_name": "t", "columns": ["content"]},
        COLUMNAR_ADD_SQL,
    ),
    "arun_auto_columnarization": ({}, RECOMMEND_SQL),
    "adefine_vector_assist_spec": (
        {"table_name": "t", "embedding_column": "embedding"},
        DEFINE_SPEC_SQL,
    ),
    "aapply_vector_assist_spec": (
        {"table_name": "t", "embedding_column": "embedding"},
        LATEST_SPEC_SQL,
    ),
    "aget_vector_assist_recommendations": (
        {"table_name": "t", "embedding_column": "embedding"},
        LATEST_SPEC_SQL,
    ),
}


def _result(scalar: Any = None, rows: Any = None, first: Any = None) -> MagicMock:
    result = MagicMock()
    result.scalar.return_value = scalar
    result.fetchall.return_value = rows or []
    result.mappings.return_value = MagicMock()
    result.mappings.return_value.__iter__.return_value = iter(rows or [])
    result.mappings.return_value.first.return_value = first
    return result


def _mapping_result(rows: list[dict]) -> MagicMock:
    result = MagicMock()
    result.mappings.return_value = rows
    return result


def _sql(call: Any) -> str:
    return str(call[0][0])


def _params(call: Any) -> Any:
    return call[0][1] if len(call[0]) > 1 else None


class TestEngineAlloyDBAIUnit:
    @pytest.fixture
    def engine(self):
        eng = AlloyDBEngine.__new__(AlloyDBEngine)
        eng._pool = MagicMock()
        eng._loop = None
        return eng

    @pytest.fixture
    def conn(self, engine):
        with patch.object(engine._pool, "connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn
            yield mock_conn

    # ai.initialize_embeddings

    @pytest.mark.asyncio
    async def test_ainitialize_embeddings_guard_passes(self, engine, conn):
        """Empty column: the guard query runs first, then the CALL runs in
        autocommit mode."""
        conn.execute.side_effect = [_result(scalar=False), MagicMock()]
        await engine.ainitialize_embeddings(
            "test_table", "test-model", "content", "embedding"
        )
        guard_call, call = conn.execute.call_args_list
        assert _sql(guard_call) == (
            'SELECT EXISTS (SELECT 1 FROM "public"."test_table" '
            'WHERE "embedding" IS NOT NULL)'
        )
        assert _sql(call) == INITIALIZE_SQL
        assert _params(call) == {
            "model_id": "test-model",
            "table_name": '"public"."test_table"',
            "content_column": "content",
            "embedding_column": "embedding",
        }
        # ai.initialize_embeddings COMMITs internally, so it must run outside
        # a transaction block (autocommit), set before the CALL.
        conn.execution_options.assert_awaited_once_with(isolation_level="AUTOCOMMIT")
        names = [c[0] for c in conn.method_calls]
        last_execute = max(i for i, n in enumerate(names) if n == "execute")
        assert names.index("execution_options") < last_execute
        conn.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ainitialize_embeddings_guard_refuses(self, engine, conn):
        """Existing vectors and overwrite=False: ValueError, and the CALL is
        never sent."""
        conn.execute.return_value = _result(scalar=True)
        with pytest.raises(ValueError, match="already contains embeddings"):
            await engine.ainitialize_embeddings(
                "test_table", "test-model", "content", "embedding"
            )
        assert conn.execute.call_count == 1
        assert "ai.initialize_embeddings" not in _sql(conn.execute.call_args)
        conn.execution_options.assert_not_called()

    @pytest.mark.asyncio
    async def test_ainitialize_embeddings_overwrite_skips_guard(self, engine, conn):
        await engine.ainitialize_embeddings(
            "test_table",
            "test-model",
            "custom_content",
            "custom_embedding",
            schema_name="myschema",
            overwrite=True,
        )
        assert conn.execute.call_count == 1
        assert _sql(conn.execute.call_args) == INITIALIZE_SQL
        assert _params(conn.execute.call_args) == {
            "model_id": "test-model",
            "table_name": '"myschema"."test_table"',
            "content_column": "custom_content",
            "embedding_column": "custom_embedding",
        }

    @pytest.mark.asyncio
    async def test_ainitialize_embeddings_guard_quotes_identifiers(self, engine, conn):
        """Identifiers in the guard query are quoted, and ':' is escaped so
        text() does not see a bind parameter."""
        conn.execute.side_effect = [_result(scalar=False), MagicMock()]
        await engine.ainitialize_embeddings(
            'we"ird:tbl', "m", "content", 'emb"; DROP TABLE x;--', schema_name="s:1"
        )
        guard_call = conn.execute.call_args_list[0]
        assert _sql(guard_call) == (
            'SELECT EXISTS (SELECT 1 FROM "s:1"."we""ird:tbl" '
            'WHERE "emb""; DROP TABLE x;--" IS NOT NULL)'
        )
        assert not guard_call[0][0]._bindparams
        assert _params(conn.execute.call_args_list[1])["table_name"] == (
            '"s:1"."we""ird:tbl"'
        )

    @pytest.mark.parametrize(
        "name",
        ["_ainitialize_embeddings", "ainitialize_embeddings", "initialize_embeddings"],
    )
    def test_initialize_embeddings_columns_are_required(self, engine, name):
        """content_column and embedding_column have no default (no fallback)."""
        params = inspect.signature(getattr(engine, name)).parameters
        assert list(params)[:4] == [
            "table_name",
            "model_id",
            "content_column",
            "embedding_column",
        ]
        for p in ("table_name", "model_id", "content_column", "embedding_column"):
            assert params[p].default is inspect.Parameter.empty
        assert params["schema_name"].default == "public"
        assert params["overwrite"].default is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"table_name": ""}, "table_name"),
            ({"model_id": " "}, "model_id"),
            ({"content_column": ""}, "content_column"),
            ({"embedding_column": None}, "embedding_column"),
            ({"schema_name": ""}, "schema_name"),
        ],
    )
    async def test_ainitialize_embeddings_rejects_empty_names(
        self, engine, conn, kwargs, field
    ):
        args = {
            "table_name": "t",
            "model_id": "m",
            "content_column": "content",
            "embedding_column": "embedding",
            **kwargs,
        }
        with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
            await engine.ainitialize_embeddings(**args)
        conn.execute.assert_not_called()

    def test_initialize_embeddings_sync(self, engine):
        engine._run_as_sync = MagicMock(side_effect=lambda coro: coro.close())
        engine.initialize_embeddings("t", "m", "c", "e", "s", True)
        engine._run_as_sync.assert_called_once()
        coro = engine._run_as_sync.call_args[0][0]
        assert coro.cr_code.co_name == "_ainitialize_embeddings"

    # Columnar engine

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine(self, engine, conn):
        conn.execute.return_value = _result(scalar=1)
        await engine.aenable_columnar_engine("test_table", ["content"])
        assert conn.execute.call_count == 1
        assert _sql(conn.execute.call_args) == COLUMNAR_ADD_SQL
        assert _params(conn.execute.call_args) == {
            "table_name": '"public"."test_table"',
            # Raw names: the server looks each one up verbatim.
            "columns": "content",
        }
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_mixed_case_column_is_not_quoted(
        self, engine, conn
    ):
        conn.execute.return_value = _result(scalar=1)
        await engine.aenable_columnar_engine(
            "T", ["MyCol", "my col", 'we"ird'], schema_name='My"Schema'
        )
        assert _params(conn.execute.call_args) == {
            "table_name": '"My""Schema"."T"',
            "columns": 'MyCol,my col,we"ird',
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["a,b", "a:1", " a", "a ", ""])
    async def test_aenable_columnar_engine_rejects_unrepresentable_names(
        self, engine, conn, bad
    ):
        with pytest.raises(ValueError, match="cannot be added to the columnar"):
            await engine.aenable_columnar_engine("t", ["content", bad])
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_default_columns(self, engine, conn):
        """Without columns, the default query excludes vector-typed and
        tracking columns (filtered by schema and table)."""
        conn.execute.side_effect = [
            _result(rows=[("id",), ("content",)]),
            _result(scalar=3),
        ]
        await engine.aenable_columnar_engine("test_table", schema_name="myschema")
        col_call, add_call = conn.execute.call_args_list
        assert _sql(col_call) == DEFAULT_COLUMNS_SQL
        assert _params(col_call) == {
            "schema_name": "myschema",
            "table_name": "test_table",
            "tracking_column_pattern": TRACKING_PATTERN,
        }
        assert _sql(add_call) == COLUMNAR_ADD_SQL
        assert _params(add_call) == {
            "table_name": '"myschema"."test_table"',
            "columns": "id,content",
        }

    def test_tracking_column_pattern_escapes_underscores(self):
        """The LIKE pattern matches the tracking columns only: '_' is escaped,
        so e.g. 'googleXmlXtrack...' does not match."""
        import re

        # Translate the LIKE pattern (default escape character '\') to a regex.
        regex = ""
        i = 0
        while i < len(TRACKING_PATTERN):
            ch = TRACKING_PATTERN[i]
            if ch == "\\":
                regex += re.escape(TRACKING_PATTERN[i + 1])
                i += 2
                continue
            regex += ".*" if ch == "%" else "." if ch == "_" else re.escape(ch)
            i += 1
        assert re.fullmatch(regex, "google_ml_track_stale_embedding_1")
        assert re.fullmatch(regex, "google_ml_track_stale_embedding_12")
        assert not re.fullmatch(regex, "googleXmlXtrackXstaleXembeddingX1")
        assert not re.fullmatch(regex, "content")

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_no_columns_found_raises(self, engine, conn):
        conn.execute.return_value = _result(rows=[])
        with pytest.raises(ValueError, match="No non-vector columns"):
            await engine.aenable_columnar_engine("t")
        assert conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_empty_list_raises(self, engine, conn):
        with pytest.raises(ValueError, match="columns must not be empty"):
            await engine.aenable_columnar_engine("t", [])
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("columns", ["content", ["content", 1], [None]])
    async def test_aenable_columnar_engine_rejects_non_list_of_str(
        self, engine, conn, columns
    ):
        """A bare str is not split into characters; entries must be str."""
        with pytest.raises(ValueError, match="must be a list of column names"):
            await engine.aenable_columnar_engine("t", columns)
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_accepts_tuple(self, engine, conn):
        conn.execute.return_value = _result(scalar=1)
        await engine.aenable_columnar_engine("t", ("id", "content"))  # type: ignore[arg-type]
        assert _params(conn.execute.call_args)["columns"] == "id,content"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs", [{"table_name": ""}, {"schema_name": " "}])
    async def test_aenable_columnar_engine_rejects_empty_names(
        self, engine, conn, kwargs
    ):
        args = {"table_name": "t", "columns": ["content"], **kwargs}
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await engine.aenable_columnar_engine(**args)
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("returned", [0, None])
    async def test_aenable_columnar_engine_warns_when_add_returns_zero(
        self, engine, conn, caplog, returned
    ):
        """google_columnar_engine_add returns 0 (after a server WARNING) instead
        of raising on its failure paths; that must be visible to the caller."""
        conn.execute.return_value = _result(scalar=returned)
        with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
            await engine.aenable_columnar_engine("test_table", ["content"])
        warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings_logged) == 1
        message = warnings_logged[0].getMessage()
        assert f"google_columnar_engine_add returned {returned!r}" in message
        assert '"public"."test_table"' in message
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aenable_columnar_engine_no_warning_on_success(
        self, engine, conn, caplog
    ):
        conn.execute.return_value = _result(scalar=12)
        with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
            await engine.aenable_columnar_engine("test_table", ["content"])
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_enable_columnar_engine_sync(self, engine):
        engine._run_as_sync = MagicMock(side_effect=lambda coro: coro.close())
        engine.enable_columnar_engine("t", ["c"], "s")
        coro = engine._run_as_sync.call_args[0][0]
        assert coro.cr_code.co_name == "_aenable_columnar_engine"

    # Auto-columnarization

    @pytest.mark.asyncio
    async def test_arun_auto_columnarization(self, engine, conn):
        conn.execute.return_value = _mapping_result(
            [{"total_size_in_mb": 7, "columns": "public.t.a,public.t.b"}]
        )
        rows = await engine.arun_auto_columnarization()
        assert rows == [{"total_size_in_mb": 7, "columns": "public.t.a,public.t.b"}]
        assert _sql(conn.execute.call_args) == RECOMMEND_SQL
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arun_auto_columnarization_no_rows(self, engine, conn):
        conn.execute.return_value = _mapping_result([])
        assert await engine.arun_auto_columnarization() == []

    def test_run_auto_columnarization_sync(self, engine):
        expected = [{"total_size_in_mb": 1, "columns": "x"}]

        def run(coro):
            coro.close()
            return expected

        engine._run_as_sync = MagicMock(side_effect=run)
        assert engine.run_auto_columnarization() == expected
        coro = engine._run_as_sync.call_args[0][0]
        assert coro.cr_code.co_name == "_arun_auto_columnarization"

    def test_enable_auto_columnarization_was_renamed(self, engine):
        assert not hasattr(engine, "enable_auto_columnarization")
        assert not hasattr(engine, "aenable_auto_columnarization")

    # Vector Assist

    @pytest.mark.asyncio
    async def test_adefine_vector_assist_spec(self, engine, conn):
        conn.execute.return_value = _mapping_result(
            [{"recommendation_id": "r1", "vector_spec_id": "spec1"}]
        )
        res = await engine.adefine_vector_assist_spec("test_table", "embedding")
        assert res == [{"recommendation_id": "r1", "vector_spec_id": "spec1"}]
        assert _sql(conn.execute.call_args) == DEFINE_SPEC_SQL
        assert _params(conn.execute.call_args) == {
            "table_name": "test_table",
            "schema_name": "public",
            "embedding_column": "embedding",
        }
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adefine_vector_assist_spec_with_embedding_model(self, engine, conn):
        conn.execute.return_value = _mapping_result([])
        await engine.adefine_vector_assist_spec(
            "test_table", "emb", "myschema", embedding_model="text-embedding-005"
        )
        assert _sql(conn.execute.call_args) == (
            "SELECT * FROM vector_assist.define_spec("
            "table_name => :table_name, schema_name => :schema_name, "
            "vector_column_name => :embedding_column, "
            "embedding_model => :embedding_model)"
        )
        assert _params(conn.execute.call_args) == {
            "table_name": "test_table",
            "schema_name": "myschema",
            "embedding_column": "emb",
            "embedding_model": "text-embedding-005",
        }

    @pytest.mark.asyncio
    async def test_aapply_vector_assist_spec_latest(self, engine, conn):
        """Without spec_id, the latest spec is resolved and applied by id."""
        conn.execute.side_effect = [
            _result(first={"spec_id": "spec_latest"}),
            _result(scalar=True),
        ]
        assert await engine.aapply_vector_assist_spec("test_table", "embedding") is True
        spec_call, apply_call = conn.execute.call_args_list
        assert _sql(spec_call) == LATEST_SPEC_SQL
        assert _params(spec_call) == {
            "table_name": "test_table",
            "schema_name": "public",
            "embedding_column": "embedding",
        }
        assert (
            _sql(apply_call) == "SELECT vector_assist.apply_spec(spec_id => :spec_id)"
        )
        assert _params(apply_call) == {"spec_id": "spec_latest"}
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aapply_and_aget_resolve_the_same_spec(self, engine, conn):
        spec = _result(first={"spec_id": "s9"})
        other = _result(scalar=True)
        conn.execute.side_effect = [spec, other, spec, other]
        await engine.aapply_vector_assist_spec("t", "e", "s")
        await engine.aget_vector_assist_recommendations("t", "e", "s")
        calls = conn.execute.call_args_list
        assert _sql(calls[0]) == _sql(calls[2]) == LATEST_SPEC_SQL
        assert _params(calls[0]) == _params(calls[2])
        assert _params(calls[1]) == _params(calls[3]) == {"spec_id": "s9"}

    @pytest.mark.asyncio
    async def test_aapply_vector_assist_spec_without_spec_raises(self, engine, conn):
        conn.execute.return_value = _result(first=None)
        with pytest.raises(ValueError, match="No Vector Assist spec found"):
            await engine.aapply_vector_assist_spec("t", "e")
        assert conn.execute.call_count == 1
        conn.commit.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spec_id", ["spec123", ""])
    async def test_aapply_vector_assist_spec_with_spec_id(self, engine, conn, spec_id):
        """An explicit spec_id (even "") is applied directly, without a lookup."""
        conn.execute.return_value = _result(scalar=False)
        assert (
            await engine.aapply_vector_assist_spec("t", "e", spec_id=spec_id) is False
        )
        assert conn.execute.call_count == 1
        assert _params(conn.execute.call_args) == {"spec_id": spec_id}

    @pytest.mark.asyncio
    async def test_aget_vector_assist_recommendations(self, engine, conn):
        conn.execute.side_effect = [
            _result(first={"spec_id": "spec123"}),
            _mapping_result([{"rec": "ok"}]),
        ]
        res = await engine.aget_vector_assist_recommendations("test_table", "embedding")
        assert res == [{"rec": "ok"}]
        spec_call, rec_call = conn.execute.call_args_list
        assert _sql(spec_call) == LATEST_SPEC_SQL
        assert _sql(rec_call) == (
            "SELECT * FROM vector_assist.get_recommendations(spec_id => :spec_id)"
        )
        assert _params(rec_call) == {"spec_id": "spec123"}

    @pytest.mark.asyncio
    async def test_aget_vector_assist_recommendations_no_spec_warns(
        self, engine, conn, caplog
    ):
        conn.execute.return_value = _result(first=None)
        with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
            assert await engine.aget_vector_assist_recommendations("t", "e") == []
        assert any(
            "No Vector Assist spec found" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_vector_assist_adversarial_names_are_bound(self, engine, conn):
        """vector_assist takes raw names as bound TEXT parameters, so injection
        payloads never reach the SQL text."""
        table = 'table"; DROP TABLE users;--'
        schema = 'public"; DROP TABLE users;--'
        column = 'embedding"; DROP TABLE users;--'
        conn.execute.side_effect = [
            _mapping_result([]),
            _result(first={"spec_id": "s1"}),
            _result(scalar=True),
            _result(first=None),
        ]
        await engine.adefine_vector_assist_spec(table, column, schema)
        await engine.aapply_vector_assist_spec(table, column, schema)
        await engine.aget_vector_assist_recommendations(table, column, schema)
        for call in conn.execute.call_args_list:
            assert "DROP TABLE users" not in _sql(call)
        expected = {
            "table_name": table,
            "schema_name": schema,
            "embedding_column": column,
        }
        calls = conn.execute.call_args_list
        assert _params(calls[0]) == expected
        assert _params(calls[1]) == expected
        assert _params(calls[3]) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name",
        [
            "adefine_vector_assist_spec",
            "aapply_vector_assist_spec",
            "aget_vector_assist_recommendations",
        ],
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"table_name": "", "embedding_column": "e"},
            {"table_name": "t", "embedding_column": ""},
            {"table_name": "t", "embedding_column": "e", "schema_name": ""},
        ],
    )
    async def test_vector_assist_rejects_empty_names(self, engine, conn, name, kwargs):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await getattr(engine, name)(**kwargs)
        conn.execute.assert_not_called()

    @pytest.mark.parametrize(
        "name, private",
        [
            ("define_vector_assist_spec", "_adefine_vector_assist_spec"),
            ("apply_vector_assist_spec", "_aapply_vector_assist_spec"),
            (
                "get_vector_assist_recommendations",
                "_aget_vector_assist_recommendations",
            ),
        ],
    )
    def test_vector_assist_sync_wrappers(self, engine, name, private):
        def run(coro):
            coro.close()
            return "sentinel"

        engine._run_as_sync = MagicMock(side_effect=run)
        assert getattr(engine, name)("t", "e") == "sentinel"
        coro = engine._run_as_sync.call_args[0][0]
        assert coro.cr_code.co_name == private

    # Missing-extension detection

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name, friendly_msg",
        [
            ("ainitialize_embeddings", "ai.initialize_embeddings is not available"),
            ("aenable_columnar_engine", "AlloyDB Columnar Engine is not installed"),
            ("arun_auto_columnarization", "AlloyDB Columnar Engine is not installed"),
            (
                "adefine_vector_assist_spec",
                "AlloyDB Vector Assist extension is not installed",
            ),
            (
                "aapply_vector_assist_spec",
                "AlloyDB Vector Assist extension is not installed",
            ),
            (
                "aget_vector_assist_recommendations",
                "AlloyDB Vector Assist extension is not installed",
            ),
        ],
    )
    async def test_missing_extension_detection_uses_sqlstate(
        self, engine, conn, method_name, friendly_msg
    ):
        """Unrelated DB errors are re-raised unchanged even though the SQL
        statement (included in str(e)) names the extension; SQLSTATE 42883
        maps to the friendly RuntimeError."""
        kwargs, statement = _EXTENSION_CALLS[method_name]
        method = getattr(engine, method_name)

        unrelated = ProgrammingError(
            statement,
            {"table_name": "t"},
            _FakeDBAPIError("permission denied for table t", "42501"),
        )
        assert statement in str(unrelated)
        conn.execute.side_effect = unrelated
        with pytest.raises(ProgrammingError) as exc_info:
            await method(**kwargs)
        assert exc_info.value is unrelated

        # Undefined function (SQLSTATE 42883); the catalog lookup fails with
        # the same error, so the SQLSTATE alone decides.
        missing = ProgrammingError(
            statement, {}, _FakeDBAPIError("function does not exist", "42883")
        )
        conn.execute.side_effect = missing
        with pytest.raises(RuntimeError, match=friendly_msg) as rt_info:
            await method(**kwargs)
        assert rt_info.value.__cause__ is missing

        # psycopg-style orig exposing only ``pgcode``.
        conn.execute.side_effect = ProgrammingError(
            statement, {}, _FakeDBAPIError("function does not exist", None, "42883")
        )
        with pytest.raises(RuntimeError, match=friendly_msg):
            await method(**kwargs)

        # Non-DBAPI errors (no ``orig``) are re-raised unchanged.
        plain = RuntimeError(
            "google_columnar_engine vector_assist ai.initialize_embeddings"
        )
        conn.execute.side_effect = plain
        with pytest.raises(RuntimeError) as rt_info:
            await method(**kwargs)
        assert rt_info.value is plain

    @pytest.mark.asyncio
    async def test_columnar_engine_disabled_error_is_not_masked(self, engine, conn):
        """A 55000 error (columnar engine flag off) propagates unchanged."""
        err = ProgrammingError(
            RECOMMEND_SQL,
            {},
            _FakeDBAPIError(
                "google_columnar_engine module must be loaded via "
                "shared_preload_libraries and google_columnar_engine.enabled "
                "must be turned on.",
                "55000",
            ),
        )
        conn.execute.side_effect = err
        with pytest.raises(ProgrammingError) as exc_info:
            await engine.arun_auto_columnarization()
        assert exc_info.value is err

    @pytest.mark.asyncio
    async def test_guard_query_error_is_not_masked(self, engine, conn):
        """An error from the overwrite guard (e.g. a missing table) is raised
        as is, not reported as a missing extension."""
        err = ProgrammingError(
            "SELECT EXISTS", {}, _FakeDBAPIError("relation does not exist", "42P01")
        )
        conn.execute.side_effect = err
        with pytest.raises(ProgrammingError) as exc_info:
            await engine.ainitialize_embeddings("t", "m", "c", "e")
        assert exc_info.value is err
        assert conn.execute.call_count == 1

    _CATALOG_CASES = [
        ("ainitialize_embeddings", "3F000", "ai", "initialize_embeddings"),
        ("aenable_columnar_engine", "42883", None, "google_columnar_engine_add"),
        (
            "arun_auto_columnarization",
            "42883",
            None,
            "google_columnar_engine_recommend",
        ),
        ("adefine_vector_assist_spec", "3F000", "vector_assist", "define_spec"),
        ("aapply_vector_assist_spec", "42P01", "vector_assist", "apply_spec"),
        (
            "aget_vector_assist_recommendations",
            "42P01",
            "vector_assist",
            "get_recommendations",
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name, sqlstate, schema, function", _CATALOG_CASES)
    @pytest.mark.parametrize("function_exists", [True, False])
    async def test_missing_object_error_checked_against_catalog(
        self, engine, conn, method_name, sqlstate, schema, function, function_exists
    ):
        """A missing-object SQLSTATE is re-raised unchanged when the catalog
        shows the extension function exists (e.g. a nonexistent user schema),
        and mapped to the friendly RuntimeError when it is absent."""
        kwargs, statement = _EXTENSION_CALLS[method_name]
        err = ProgrammingError(
            statement, {}, _FakeDBAPIError("some object does not exist", sqlstate)
        )
        conn.execute.side_effect = [err, _result(scalar=function_exists)]
        if function_exists:
            with pytest.raises(ProgrammingError) as exc_info:
                await getattr(engine, method_name)(**kwargs)
            assert exc_info.value is err
        else:
            with pytest.raises(RuntimeError) as rt_info:
                await getattr(engine, method_name)(**kwargs)
            assert rt_info.value.__cause__ is err
        catalog_call = conn.execute.call_args_list[-1]
        assert "pg_catalog.pg_proc" in _sql(catalog_call)
        assert _params(catalog_call) == {"schema": schema, "name": function}
