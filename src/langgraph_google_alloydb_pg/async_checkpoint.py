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

import asyncio
import asyncpg # type: ignore

from contextlib import asynccontextmanager

import json
from typing import List, Sequence, Any, AsyncIterator, Iterator, Optional, cast, Dict, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

from langgraph.checkpoint.serde.base import SerializerProtocol

MetadataInput = Optional[dict[str, Any]]

from .engine import (
    CHECKPOINTS_TABLE,
    CHECKPOINT_WRITES_TABLE,
    AlloyDBEngine
)


class AsyncAlloyDBSaver(BaseCheckpointSaver[str]):
    """Checkpoint stored in an AlloyDB for PostgreSQL database."""
    
    __create_key = object()

    jsonplus_serde = JsonPlusSerializer()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        super().__init__(serde=serde)
        if key != AsyncAlloyDBSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        self.schema_name = schema_name
        
    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None
    ) -> "AsyncAlloyDBSaver":
        """Create a new AsyncAlloyDBSaver instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AsyncAlloyDBSaver: A newly created instance of AsyncAlloyDBSaver.
        """
        
        checkpoints_table_schema = await engine._aload_table_schema(CHECKPOINTS_TABLE, schema_name)
        checkpoints_column_names = checkpoints_table_schema.columns.keys()
        
        checkpoints_required_columns = ["thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
            "v",
            "type",
            "checkpoint",
            "metadata"]
        
        if not (all(x in checkpoints_column_names for x in checkpoints_required_columns)):
            raise IndexError(
                f"Table checkpoints.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoints_column_names}' but required column names "
                f"'{checkpoints_required_columns}'.\nPlease create table with following schema:"
                f"\nCREATE TABLE {schema_name}.checkpoints ("
                "\n    thread_id TEXT NOT NULL,"
                "\n    checkpoint_ns TEXT NOT NULL,"
                "\n    checkpoint_id UUID NOT NULL,"
                "\n    parent_checkpoint_id UUID,"
                "\n    v INT NOT NULL,"
                "\n    type TEXT NOT NULL,"
                "\n    checkpoint JSONB NOT NULL,"
                "\n    metadata JSONB"
                "\n);"
            )
            
        checkpoint_writes_table_schema = await engine._aload_table_schema(CHECKPOINT_WRITES_TABLE, schema_name)
        checkpoint_writes_column_names = checkpoint_writes_table_schema.columns.keys()
        
        checkpoint_writes_columns = ["thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "channel",
            "type",
            "blob"]
        
        if not (all(x in checkpoint_writes_column_names for x in checkpoint_writes_columns)):
            raise IndexError(
                f"Table checkpoint_writes.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoint_writes_column_names}' but required column names "
                f"'{checkpoint_writes_columns}'.\nPlease create table with following schema:"
                f"\nCREATE TABLE {schema_name}.checkpoint_writes ("
                "\n    thread_id TEXT NOT NULL,"
                "\n    checkpoint_ns TEXT NOT NULL,"
                "\n    checkpoint_id UUID NOT NULL,"
                "\n    task_id UUID NOT NULL,"
                "\n    idx INT NOT NULL,"
                "\n    channel TEXT NOT NULL,"
                "\n    type TEXT NOT NULL,"
                "\n    blob JSONB NOT NULL"
                "\n);"
            )
        return cls(cls.__create_key, engine._pool, schema_name, serde)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.
        """
        
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s ")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(json.dumps(filter)) 

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))
        
        where, args = (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

        # Check channel values on select varible
        select = f"""
        select
            thread_id,
            checkpoint,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            metadata,
            (
                select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
                from jsonb_each_text(checkpoint -> 'channel_versions')
                inner join checkpoint_blobs bl
                    on bl.thread_id = checkpoints.thread_id
                    and bl.checkpoint_ns = checkpoints.checkpoint_ns
                    and bl.channel = jsonb_each_text.key
                    and bl.version = jsonb_each_text.value
            ) as channel_values,
            (
                select
                array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
                from checkpoint_writes cw
                where cw.thread_id = checkpoints.thread_id
                    and cw.checkpoint_ns = checkpoints.checkpoint_ns
                    and cw.checkpoint_id = checkpoints.checkpoint_id
            ) as pending_writes,
            (
                select array_agg(array[cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
                from checkpoint_writes cw
                where cw.thread_id = checkpoints.thread_id
                    and cw.checkpoint_ns = checkpoints.checkpoint_ns
                    and cw.checkpoint_id = checkpoints.parent_checkpoint_id
                    and cw.channel = '{TASKS}'
            ) as pending_sends
        from checkpoints """

        query = select + where + " ORDER BY checkpoint_id DESC"

        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.connect() as conn:
            result = await conn.stream(text(query), args)
            async for row_value in result:
                value = dict(row_value._mapping)
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    {
                        "id": value["checkpoint"].get("id"),
                        "ts": value["checkpoint"].get("ts"),
                        "v": value["checkpoint"].get("v"),
                        "channel_versions": value["checkpoint"].get("channel_versions"),
                        "versions_seen": value["checkpoint"].get("versions_seen"),
                        "pending_sends": [
                            self.serde.loads_typed((c.decode(), b)) for c, b in value["pending_sends"] or []
                        ],
                        "channel_values": if not 
                        
                        {
                            k.decode(): self.serde.loads_typed((t.decode(), v))
                            for k, t, v in value["pending_sends"]
                            if t.decode() != "empty"
                        },
                    },
                    self.jsonplus_serde.dumps(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": value["checkpoint_ns"],
                                "checkpoint_id": value["checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    [
                        (
                            tid.decode()
                        )
                        for tid, channel, t, v in value["pending_writes"]
                    ]


                )



    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.
        """
        raise NotImplementedError
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        blobs = f"""INSERT INTO "{self.schema_name}".checkpoints(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata, channel, version)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id, :checkpoint, :metadata, :channel, :version)
                    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
                """
        
        params = [
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": None,
                "parent_checkpoint_id": None,
                "checkpoint": None,
                "metadata": None,
                "channel": k,
                "version": cast(str,ver)
            }
                for k, ver in new_versions.items()
        ]

        async with self.pool.connect() as conn:
            await conn.execute(
                text(blobs),
                params,
            )
            await conn.commit()

        query = f"""INSERT INTO "{self.schema_name}".checkpoints(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata, channel, version)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id, :checkpoint, :metadata, :channel, :version)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                    DO UPDATE SET
                        checkpoint = EXCLUDED.checkpoint,
                        metadata = EXCLUDED.metadata;
            """

        async with self.pool.connect() as conn:
            await conn.execute(
                text(query),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": checkpoint_id,
                    "checkpoint": json.dumps(copy),
                    "metadata": json.dumps(dict(metadata)),
                    "channel": None,
                    "version": None,
                },
            )
            await conn.commit()

        return next_config

    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
        
            Returns:
                None
        """
        upsert = f"""INSERT INTO "{self.schema_name}".checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :type, :blob)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                    channel = EXCLUDED.channel,
                        type = EXCLUDED.type,
                        blob = EXCLUDED.blob;
                """
        insert = f"""INSERT INTO "{self.schema_name}".checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :type, :blob)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
                """
        query = upsert if all(w[0] in WRITES_IDX_MAP for w in writes) else insert

        params = [
            {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"]["checkpoint_ns"],
                "checkpoint_id": config["configurable"]["checkpoint_id"],
                "task_id": task_id,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": self.serde.dumps_typed(value)[0],
                "blob": self.serde.dumps_typed(value)[1]
            }
                for idx, (channel, value) in enumerate(writes)
        ]

        async with self.pool.connect() as conn:
            await conn.execute(
                text(query),
                params,
            )
            await conn.commit()

    
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """list checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Method impletented in AsyncAlloyDBSaver.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
        
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

        Raises:
            NotImplementedError: Method impletented in AsyncAlloyDBSaver.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
        
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Method impletented in AsyncAlloyDBSaver.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.

        Raises:
            NotImplementedError: Method impletented in AsyncAlloyDBSaver.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )