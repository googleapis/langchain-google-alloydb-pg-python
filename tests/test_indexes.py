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

import pytest

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
        index = ScaNNIndex(name="test_index", mode="auto")
        assert index.index_type == "ScaNN"
        assert index.mode == "AUTO"
        assert index.num_leaves == 5  # retained, but ignored for AUTO
        assert index.index_options() == "(mode = 'AUTO')"

    def test_scann_index_manual_mode(self):
        index = ScaNNIndex(name="test_index", mode="MANUAL", num_leaves=10)
        assert index.mode == "MANUAL"
        assert (
            index.index_options()
            == "(mode = 'MANUAL', num_leaves = 10, quantizer = sq8)"
        )

    def test_scann_index_default_mode_options_unchanged(self):
        assert (
            ScaNNIndex(num_leaves=10).index_options()
            == "(num_leaves = 10, quantizer = sq8)"
        )

    def test_scann_index_positional_args_backward_compatible(self):
        index = ScaNNIndex(
            "idx", "ScaNN", DistanceStrategy.EUCLIDEAN, None, "alloydb_scann", 42
        )
        assert index.num_leaves == 42
        assert index.mode is None

    def test_scann_index_mode_is_keyword_only(self):
        with pytest.raises(TypeError):
            ScaNNIndex(  # type: ignore[misc]
                "idx",
                "ScaNN",
                DistanceStrategy.EUCLIDEAN,
                None,
                "alloydb_scann",
                5,
                "AUTO",
            )

    @pytest.mark.parametrize("mode", ["INVALID", "", 1])
    def test_scann_index_invalid_mode(self, mode):
        with pytest.raises(ValueError, match="Invalid mode"):
            ScaNNIndex(name="test_index", mode=mode)

    @pytest.mark.parametrize("num_leaves", [0, -5, True, False, 5.5, "5", 2**31])
    def test_scann_index_num_leaves_validation(self, num_leaves):
        with pytest.raises(ValueError, match="num_leaves must be an integer"):
            ScaNNIndex(num_leaves=num_leaves)

    def test_scann_index_num_leaves_max_allowed(self):
        assert ScaNNIndex(num_leaves=2**31 - 1).num_leaves == 2**31 - 1

    def test_scann_index_calls_base_post_init(self):
        with pytest.raises(ValueError):
            ScaNNIndex(extension_name="bad name;")

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

    def test_scann_query_options_num_leaves_zero_allowed(self):
        options = ScaNNQueryOptions(num_leaves_to_search=0)
        assert options.to_parameter()[0] == "scann.num_leaves_to_search = 0"

    @pytest.mark.parametrize("value", [-1, 2**31, True, 1.5])
    def test_scann_query_options_num_leaves_invalid(self, value):
        with pytest.raises(ValueError, match="num_leaves_to_search must be an integer"):
            ScaNNQueryOptions(num_leaves_to_search=value)

    @pytest.mark.parametrize("pct", [0, 0.5, 2.5, 50, 100])
    def test_scann_query_options_pct_valid(self, pct):
        options = ScaNNQueryOptions(
            pre_reordering_num_neighbors=10, pct_leaves_to_search=pct
        )
        assert options.to_parameter() == [
            "scann.num_leaves_to_search = 1",
            "scann.pre_reordering_num_neighbors = 10",
            f"scann.pct_leaves_to_search = {pct}",
        ]

    @pytest.mark.parametrize("pct", [-0.1, 100.1])
    def test_scann_query_options_pct_out_of_range(self, pct):
        with pytest.raises(ValueError, match="between 0 and 100"):
            ScaNNQueryOptions(pct_leaves_to_search=pct)

    @pytest.mark.parametrize("pct", [True, "10"])
    def test_scann_query_options_pct_wrong_type(self, pct):
        with pytest.raises(TypeError):
            ScaNNQueryOptions(pct_leaves_to_search=pct)

    def test_scann_query_options_pct_to_string(self):
        options = ScaNNQueryOptions(num_leaves_to_search=5, pct_leaves_to_search=20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert options.to_string() == (
                "scann.num_leaves_to_search = 5, "
                "scann.pre_reordering_num_neighbors = -1, "
                "scann.pct_leaves_to_search = 20"
            )
            assert len(w) == 1
