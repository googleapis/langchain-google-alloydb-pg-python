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

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from .async_chat_message_history import AsyncAlloyDBChatMessageHistory
from .engine import AlloyDBEngine


class AlloyDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an AlloyDB for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
        history: AsyncAlloyDBChatMessageHistory,
    ):
        """AlloyDBChatMessageHistory constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (AlloyDBEngine): Database connection pool.
            history (AsyncAlloyDBChatMessageHistory): Async only implementation.

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AlloyDBChatMessageHistory.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.__history = history

    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> AlloyDBChatMessageHistory:
        """Create a new AlloyDBChatMessageHistory instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str): The schema name where the table is located (default: "public").

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AlloyDBChatMessageHistory: A newly created instance of AlloyDBChatMessageHistory.
        """
        coro = AsyncAlloyDBChatMessageHistory.create(
            engine, session_id, table_name, schema_name
        )
        history = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, history)

    @classmethod
    def create_sync(
        cls,
        engine: AlloyDBEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> AlloyDBChatMessageHistory:
        """Create a new AlloyDBChatMessageHistory instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name: The schema name where the table is located (default: "public").

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AlloyDBChatMessageHistory: A newly created instance of AlloyDBChatMessageHistory.
        """
        coro = AsyncAlloyDBChatMessageHistory.create(
            engine, session_id, table_name, schema_name
        )
        history = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, history)

    @property
    def messages(self) -> list[BaseMessage]:
        """Fetches all messages stored in AlloyDB."""
        return self._engine._run_as_sync(self.__history._aget_messages())

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """Clear the stored messages and appends a list of messages to the record in AlloyDB."""
        self.clear()
        self.add_messages(value)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in AlloyDB"""
        await self._engine._run_as_async(self.__history.aadd_message(message))

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in AlloyDB"""
        self._engine._run_as_sync(self.__history.aadd_message(message))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in AlloyDB"""
        await self._engine._run_as_async(self.__history.aadd_messages(messages))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in AlloyDB"""
        self._engine._run_as_sync(self.__history.aadd_messages(messages))

    async def aclear(self) -> None:
        """Clear session memory from AlloyDB"""
        await self._engine._run_as_async(self.__history.aclear())

    def clear(self) -> None:
        """Clear session memory from AlloyDB"""
        self._engine._run_as_sync(self.__history.aclear())
