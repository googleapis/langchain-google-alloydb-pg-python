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

from langchain_google_alloydb_pg.indexes import (  # type: ignore
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

    def test_scann_index_auto_mode(self):
        index = ScaNNIndex(name="test_index", mode="AUTO")
        assert index.index_type == "ScaNN"
        assert index.mode == "AUTO"
        assert index.index_options() == "(mode = 'AUTO')"

    def test_scann_index_invalid_mode(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid mode 'INVALID'"):
            ScaNNIndex(name="test_index", mode="INVALID")

    def test_scann_index_num_leaves_validation(self):
        import pytest

        # Test bool (True/False)
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves=True)
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves=False)

        # Test float
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves=5.5)

        # Test str
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves="5")

        # Test 0
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves=0)

        # Test negative int
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(num_leaves=-5)

        # Test mode="AUTO" with negative num_leaves (should fail because num_leaves is validated before mode="AUTO" is applied)
        with pytest.raises(ValueError, match="num_leaves must be a positive integer."):
            ScaNNIndex(mode="AUTO", num_leaves=-5)

        # Test num_leaves > 2_147_483_647 (should fail)
        with pytest.raises(
            ValueError, match="num_leaves exceeds maximum 32-bit integer limit"
        ):
            ScaNNIndex(num_leaves=3_000_000_000)

    def test_scann_query_options_num_leaves_overflow(self):
        import pytest

        with pytest.raises(
            ValueError,
            match="num_leaves_to_search exceeds maximum 32-bit integer limit",
        ):
            ScaNNQueryOptions(num_leaves_to_search=2_147_483_648)

    def test_scann_index_functions(self):
        idx_l2 = ScaNNIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        assert idx_l2.get_index_function() == "l2"
        idx_cos = ScaNNIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE)
        assert idx_cos.get_index_function() == "cosine"
        idx_dot = ScaNNIndex(distance_strategy=DistanceStrategy.INNER_PRODUCT)
        assert idx_dot.get_index_function() == "dot_prod"

    def test_scann_query_options_default(self):
        options = ScaNNQueryOptions()
        assert options.to_parameter() == [
            "scann.num_leaves_to_search = 1",
            "scann.pre_reordering_num_neighbors = -1",
        ]

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

    def test_scann_query_options_pct_leaves(self):
        options = ScaNNQueryOptions(
            pre_reordering_num_neighbors=10,
            pct_leaves_to_search=0.2,
        )
        assert options.to_parameter() == [
            "scann.pct_leaves_to_search = 0.2",
            "scann.pre_reordering_num_neighbors = 10",
        ]
        with warnings.catch_warnings(record=True) as w:
            to_str = options.to_string()
            assert (
                to_str
                == "scann.pct_leaves_to_search = 0.2, scann.pre_reordering_num_neighbors = 10"
            )

    def test_scann_query_options_both_params_warns(self):
        options = ScaNNQueryOptions(
            num_leaves_to_search=5,
            pre_reordering_num_neighbors=10,
            pct_leaves_to_search=0.5,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = options.to_parameter()
            assert len(w) == 1
            assert (
                "Both 'pct_leaves_to_search' and 'num_leaves_to_search' were provided"
                in str(w[-1].message)
            )
            assert params == [
                "scann.pct_leaves_to_search = 0.5",
                "scann.pre_reordering_num_neighbors = 10",
            ]
