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

import numbers
import warnings
from dataclasses import dataclass, field
from typing import Optional

from langchain_postgres.v2.indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFFlatQueryOptions,
    QueryOptions,
    StrategyMixin,
)

# PostgreSQL int4 upper bound; ScaNN integer options are int4 on the server.
_MAX_INT32 = 2**31 - 1
_SCANN_MODES = ("AUTO", "MANUAL")


def _is_int(value: object) -> bool:
    """True for integers (including numpy integers), excluding bools."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass
class IVFIndex(BaseIndex):
    index_type: str = "ivf"
    lists: int = 100
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(lists = {self.lists}, quantizer = {self.quantizer})"


@dataclass
class IVFQueryOptions(QueryOptions):
    probes: int = 1

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"ivf.probes = {self.probes}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"ivf.probes = {self.probes}"


@dataclass
class ScaNNIndex(BaseIndex):
    """ScaNN index configuration for AlloyDB.

    Args:
        num_leaves (int): Number of partitions. Used when ``mode`` is ``None``
            or ``"MANUAL"``; ignored when ``mode="AUTO"``. Defaults to 5.
        mode (Optional[str]): Keyword-only. ``"AUTO"`` creates an automatically
            tuned index (the server chooses the number of leaves; the table
            must contain at least 10,000 rows); ``"MANUAL"``
            creates a manually tuned index using ``num_leaves``. Defaults to
            ``None``, which emits the same options as previous releases.
    """

    index_type: str = "ScaNN"
    num_leaves: int = 5
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"
    extension_name: str = "alloydb_scann"
    # kw_only keeps the positional order of all existing fields unchanged.
    mode: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _is_int(self.num_leaves) or not 1 <= self.num_leaves <= _MAX_INT32:
            raise ValueError(
                f"num_leaves must be an integer between 1 and {_MAX_INT32}."
            )
        if self.mode is not None:
            if not isinstance(self.mode, str) or self.mode.upper() not in _SCANN_MODES:
                raise ValueError(
                    f"Invalid mode {self.mode!r}. Supported modes are 'AUTO' and 'MANUAL'."
                )
            self.mode = self.mode.upper()

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        if self.mode == "AUTO":
            return "(mode = 'AUTO')"
        if self.mode == "MANUAL":
            return f"(mode = 'MANUAL', num_leaves = {self.num_leaves}, quantizer = {self.quantizer})"
        return f"(num_leaves = {self.num_leaves}, quantizer = {self.quantizer})"

    def get_index_function(self) -> str:
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "l2"
        elif self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return "cosine"
        else:
            return "dot_prod"


@dataclass
class ScaNNQueryOptions(QueryOptions):
    """Query options for ScaNN index.

    Args:
        num_leaves_to_search (int): Number of leaves to search. ``0`` lets the
            server choose. Defaults to 1.
        pre_reordering_num_neighbors (int): Defaults to -1.
        pct_leaves_to_search (Optional[float]): Percentage (0-100, fractional
            values allowed) of leaves to search. When set, it is sent in addition to
            ``num_leaves_to_search``; the server uses the percentage and falls
            back to ``num_leaves_to_search`` if the percentage resolves to zero
            leaves. Defaults to ``None`` (not sent).
    """

    num_leaves_to_search: int = 1
    pre_reordering_num_neighbors: int = -1
    pct_leaves_to_search: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            not _is_int(self.num_leaves_to_search)
            or not 0 <= self.num_leaves_to_search <= _MAX_INT32
        ):
            raise ValueError(
                f"num_leaves_to_search must be an integer between 0 and {_MAX_INT32}."
            )
        if self.pct_leaves_to_search is not None:
            if isinstance(self.pct_leaves_to_search, bool) or not isinstance(
                self.pct_leaves_to_search, (int, float)
            ):
                raise TypeError(
                    "pct_leaves_to_search must be a number between 0 and 100."
                )
            if not 0 <= self.pct_leaves_to_search <= 100:
                raise ValueError("pct_leaves_to_search must be between 0 and 100.")

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        params = [
            f"scann.num_leaves_to_search = {self.num_leaves_to_search}",
            f"scann.pre_reordering_num_neighbors = {self.pre_reordering_num_neighbors}",
        ]
        if self.pct_leaves_to_search is not None:
            params.append(f"scann.pct_leaves_to_search = {self.pct_leaves_to_search}")
        return params

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return ", ".join(self.to_parameter())
