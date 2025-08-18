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

import warnings
from dataclasses import dataclass, field

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
    index_type: str = "ScaNN"
    num_leaves: int = 5
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"
    extension_name: str = "alloydb_scann"

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
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
    num_leaves_to_search: int = 1
    pre_reordering_num_neighbors: int = -1

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [
            f"scann.num_leaves_to_search = {self.num_leaves_to_search}",
            f"scann.pre_reordering_num_neighbors = {self.pre_reordering_num_neighbors}",
        ]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"scann.num_leaves_to_search = {self.num_leaves_to_search}, scann.pre_reordering_num_neighbors = {self.pre_reordering_num_neighbors}"
