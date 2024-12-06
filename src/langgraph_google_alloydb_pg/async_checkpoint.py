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
from contextlib import asynccontextmanager

import json
from typing import List, Sequence, Any, AsyncIterator, Iterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

from langgraph.checkpoint.serde.base import SerializerProtocol

MetadataInput = Optional[dict[str, Any]]

from .engine import AlloyDBEngine


class AsyncAlloyDBSaver(BaseCheckpointSaver[str]):
    """Checkpoint stored in an AlloyDB for PostgreSQL database."""
    
    __create_key = object()
    
    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        super().__init__(serde=serde)
        if key != AsyncAlloyDBSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        
    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        serde: Optional[SerializerProtocol] = None
    ) -> "AsyncAlloyDBSaver":
        pass
    
    
        
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        pass
    
    async def aget_tuple(self):
        pass
    
    async def aput(self):
        pass
    
    async def aput_writes(self):
        pass
    
    def list(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
        
    def get_tuple(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
        
    def put(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )
    
    def put_writes(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBSaver. Use AlloyDBSaver interface instead."
        )