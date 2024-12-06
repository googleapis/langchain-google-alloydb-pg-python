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

from collections.abc import Iterator, AsyncIterator, Sequence
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from .async_checkpoint import AsyncAlloyDBSaver
from .engine import AlloyDBEngine

class AlloyDBSaver(BaseCheckpointSaver[str]):
    
    __create_key = object()
    
    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
        checkpoint: AsyncAlloyDBSaver,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        super().__init__(serde=serde)
        if key != AlloyDBSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self._engine = engine
        self.__checkpoint = checkpoint
        
    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None
    ) -> "AlloyDBSaver":
        """Create a new AlloyDBSaver instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AlloyDBSaver: A newly created instance of AlloyDBSaver.
        """
        coro = AsyncAlloyDBSaver.create(
            engine, table_name, schema_name, serde
        )
        checkpoint = engine._run_as_async(coro)
        return cls(cls.__create_key, engine, table_name, schema_name, serde)  
    
    @classmethod
    def create_sync(
        cls,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None
    ) -> "AlloyDBSaver":
        """Create a new AlloyDBSaver instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            table_name (str): Table name that stores the chat message history.
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AlloyDBSaver: A newly created instance of AlloyDBSaver.
        """
        coro = AsyncAlloyDBSaver.create(
            engine, serde
        )
        checkpoint = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, table_name, schema_name, serde)  
    
    async def alist(self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None
    ) -> AsyncIterator[CheckpointTuple]:
        '''List checkpoints from AlloyDB '''
        await self._engine._run_as_async(self.__checkpoint.alist(config, filter, before, limit))
    
    def list(self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        '''List checkpoints from AlloyDB '''
        self._engine._run_as_sync(self.__checkpoint.alist(config, filter, before, limit))
        
    async def aget_tuple(
        self,
        config: RunnableConfig
    ) -> Optional[CheckpointTuple]:
        '''Get a checkpoint tuple from AlloyDB'''
        await self._engine._run_as_sync(self.__checkpoint.aget_tuple(config))
    
    def get_tuple(
        self,
        config: RunnableConfig
    ) -> Optional[CheckpointTuple]:
        '''Get a checkpoint tuple from AlloyDB'''
        self._engine._run_as_sync(self.__checkpoint.aget_tuple(config))
        
    async def aput(self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions
    ) -> RunnableConfig:
        '''Save a checkpoint to AlloyDB'''
        await self._engine._run_as_async(self.__checkpoint.aput(config, checkpoint, metadata, new_versions))
    
    def put(self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions
    ) -> RunnableConfig:
        '''Save a checkpoint to AlloyDB'''
        self._engine._run_as_sync(self.__checkpoint.aput(config, checkpoint, metadata, new_versions))
        
    async def aput_writes(self,
                config: RunnableConfig,
                writes: Sequence[tuple[str, Any]],
                task_id: str
    ) -> None:
        '''Store intermediate writes linked to a checkpoint'''
        await self._engine._run_as_sync(self.__checkpoint.aput_writes(config, writes, task_id))
    
    def put_writes(self,
                config: RunnableConfig,
                writes: Sequence[tuple[str, Any]],
                task_id: str
    ) -> None:
        '''Store intermediate writes linked to a checkpoint'''
        self._engine._run_as_sync(self.__checkpoint.aput_writes(config, writes, task_id))
        