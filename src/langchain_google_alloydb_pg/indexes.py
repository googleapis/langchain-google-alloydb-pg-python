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

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar


@dataclass
class StrategyMixin:
    operator: str
    search_function: str
    index_function: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the Distance strategies."""

    PGVECTOR_EUCLIDEAN = "<->", "l2_distance", "vector_l2_ops"
    PGVECTOR_COSINE_DISTANCE = "<=>", "cosine_distance", "vector_cosine_ops"
    PGVECTOR_INNER_PRODUCT = "<#>", "ip_distance", "vector_ip_ops"
    # For SCANN index only
    SCANN_EUCLIDEAN = "<->", "l2_distance", "l2"
    SCANN_COSINE_DISTANCE = "<=>", "cosine_distance", "cosine"
    SCANN_INNER_PRODUCT = "<#>", "ip_distance", "dot_product"


DEFAULT_DISTANCE_STRATEGY: DistanceStrategy = DistanceStrategy.PGVECTOR_COSINE_DISTANCE
DEFAULT_INDEX_NAME: str = "langchainvectorindex"


@dataclass
class BaseIndex(ABC):
    name: str = DEFAULT_INDEX_NAME
    index_type: str = "base"
    distance_strategy: DistanceStrategy = field(
        default_factory=lambda: DistanceStrategy.PGVECTOR_COSINE_DISTANCE
    )
    partial_indexes: Optional[List[str]] = None

    @abstractmethod
    def index_options(self) -> str:
        raise NotImplementedError(
            "index_options method must be implemented by subclass"
        )


@dataclass
class ExactNearestNeighbor(BaseIndex):
    index_type: str = "exactnearestneighbor"


@dataclass
class HNSWIndex(BaseIndex):
    index_type: str = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def index_options(self) -> str:
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"


@dataclass
class QueryOptions(ABC):
    @abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError("to_string method must be implemented by subclass")


@dataclass
class HNSWQueryOptions(QueryOptions):
    ef_search: int = 40

    def to_string(self) -> str:
        return f"hnsw.ef_search = {self.ef_search}"


@dataclass
class IVFFlatIndex(BaseIndex):
    index_type: str = "ivfflat"
    lists: int = 1

    def index_options(self) -> str:
        return f"(lists = {self.lists})"


@dataclass
class IVFFlatQueryOptions(QueryOptions):
    probes: int = 1

    def to_string(self) -> str:
        return f"ivfflat.probes = {self.probes}"


@dataclass
class IVFIndex(BaseIndex):
    index_type: str = "ivf"
    lists: int = 100
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"

    def index_options(self) -> str:
        return f"(lists = {self.lists}, quantizer = {self.quantizer})"


@dataclass
class IVFQueryOptions(QueryOptions):
    probes: int = 1

    def to_string(self) -> str:
        return f"ivf.probes = {self.probes}"


@dataclass
class SCANNIndex(BaseIndex):
    distance_strategy: DistanceStrategy = field(
        default_factory=lambda: DistanceStrategy.SCANN_COSINE_DISTANCE
    )
    index_type: str = "scann"
    num_leaves: int = 5
    quantizer: str = field(
        default="sq8", init=False
    )  # Excludes `quantizer` from initialization currently only supports the value "sq8"

    def index_options(self) -> str:
        return f"(num_leaves = {self.num_leaves}, quantizer = {self.quantizer})"
