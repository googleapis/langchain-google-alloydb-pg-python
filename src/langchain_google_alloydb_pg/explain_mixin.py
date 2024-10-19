from typing import Any, Awaitable, Callable, Sequence

from sqlalchemy import RowMapping


class ExplainMixin:
    """
    Mixin class to provide explain plan functionality for database queries.
    """

    _explain_enabled: bool = False

    async def explain(
        self,
        query_method: Callable[..., Awaitable[Sequence[RowMapping]]],
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[RowMapping]:
        """
        Provides the query plan for the given query method without executing it.

        Args:
            query_method: The method that executes the database query
                            (e.g., `asimilarity_search`).
            *args: Positional arguments to pass to the query method.
            *kwargs: Keyword arguments to pass to the query method.

        Returns:
            The query plan.
        """
        self._explain_enabled = True
        try:
            return await query_method(*args, **kwargs)
        finally:
            self._explain_enabled = False
