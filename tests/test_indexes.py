# Copyright 2025 Google LLC
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

from langchain_google_alloydb_pg.indexes import (
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFFlatQueryOptions,
    IVFIndex,
    IVFQueryOptions,
    ScaNNIndex,
    ScaNNQueryOptions,
)


class TestAlloyDBIndex:
    def test_distance_strategy(self):
        assert DistanceStrategy.EUCLIDEAN.operator == "<->"
        assert DistanceStrategy.EUCLIDEAN.search_function == "l2_distance"
        assert DistanceStrategy.EUCLIDEAN.index_function == "vector_l2_ops"

        assert DistanceStrategy.COSINE_DISTANCE.operator == "<=>"
        assert DistanceStrategy.COSINE_DISTANCE.search_function == "cosine_distance"
        assert DistanceStrategy.COSINE_DISTANCE.index_function == "vector_cosine_ops"

        assert DistanceStrategy.INNER_PRODUCT.operator == "<#>"
        assert DistanceStrategy.INNER_PRODUCT.search_function == "inner_product"
        assert DistanceStrategy.INNER_PRODUCT.index_function == "vector_ip_ops"

        scann_index = ScaNNIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        assert scann_index.get_index_function() == "l2"
        scann_index = ScaNNIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE)
        assert scann_index.get_index_function() == "cosine"
        scann_index = ScaNNIndex(distance_strategy=DistanceStrategy.INNER_PRODUCT)
        assert scann_index.get_index_function() == "dot_prod"

    def test_hnsw_index(self):
        index = HNSWIndex(name="test_index", m=32, ef_construction=128)
        assert index.index_type == "hnsw"
        assert index.m == 32
        assert index.ef_construction == 128
        assert index.index_options() == "(m = 32, ef_construction = 128)"

    def test_hnsw_query_options(self):
        options = HNSWQueryOptions(ef_search=80)
        assert options.to_parameter() == ["hnsw.ef_search = 80"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()

            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )

    def test_ivfflat_index(self):
        index = IVFFlatIndex(name="test_index", lists=200)
        assert index.index_type == "ivfflat"
        assert index.lists == 200
        assert index.index_options() == "(lists = 200)"

    def test_ivfflat_query_options(self):
        options = IVFFlatQueryOptions(probes=2)
        assert options.to_parameter() == ["ivfflat.probes = 2"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()
            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )

    def test_ivf_index(self):
        index = IVFIndex(name="test_index", lists=200)
        assert index.index_type == "ivf"
        assert index.lists == 200
        assert index.quantizer == "sq8"  # Check default value
        assert index.index_options() == "(lists = 200, quantizer = sq8)"

    def test_ivf_query_options(self):
        options = IVFQueryOptions(probes=2)
        assert options.to_parameter() == ["ivf.probes = 2"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()
            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )

    def test_scann_index(self):
        index = ScaNNIndex(name="test_index", num_leaves=10)
        assert index.index_type == "ScaNN"
        assert index.num_leaves == 10
        assert index.quantizer == "sq8"  # Check default value
        assert index.index_options() == "(num_leaves = 10, quantizer = sq8)"

    def test_scann_query_options(self):
        options = ScaNNQueryOptions(
            num_leaves_to_search=2, pre_reordering_num_neighbors=10
        )
        assert options.to_parameter() == [
            "scann.num_leaves_to_search = 2",
            "scann.pre_reordering_num_neighbors = 10",
        ]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()
            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )
