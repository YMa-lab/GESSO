from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure


@pytest.fixture
def gas_report(gesso_model_no_genesets, genesets_dict):
    return gesso_model_no_genesets.compute_gas(
        genesets_dict=genesets_dict,
        n_jobs=1,
    )


class TestGeneSetActivityScoresReport:
    def test_gas_df_shape(self, gas_report, genesets_dict):
        df = gas_report.gas_df()
        assert df.shape == (60, len(genesets_dict))
        assert set(df.columns) == set(genesets_dict.keys())

    def test_locations_df(self, gas_report, locations_df):
        df = gas_report.locations_df()
        assert list(df.columns) == ["x", "y"]
        pd.testing.assert_index_equal(df.index, locations_df.index)

    def test_gene_contributions_df_sorted_by_weight(self, gas_report):
        m = gas_report.gene_contributions_df("geneset_A", sort_by="gene_contribution")
        col = m["geneset_A"].to_numpy()

        assert np.all(col[:-1] >= col[1:])

    def test_gene_contributions_df_sorted_by_gene_name(self, gas_report):
        m = gas_report.gene_contributions_df("geneset_A", sort_by="gene_name")
        names = m.index.to_list()
        assert names == sorted(names)

    def test_gene_contributions_df_invalid_sort(self, gas_report):
        with pytest.raises(ValueError, match="sort_by"):
            gas_report.gene_contributions_df("geneset_A", sort_by="bogus")

    def test_plot_gas_spatial_map_returns_figure(self, gas_report):
        fig = gas_report.plot_gas_spatial_map("geneset_A")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_gas_with_user_axes(self, gas_report):
        _, ax = plt.subplots()
        fig = gas_report.plot_gas_spatial_map("geneset_A", ax=ax)
        assert isinstance(fig, Figure)
        plt.close("all")


@pytest.fixture
def htest_report(gesso_model, genesets_dict):
    return gesso_model.htest_elevated_gas(
        geneset="geneset_A",
        n_permutations=15,
        n_jobs=1,
    )


class TestPermutationTestReport:
    def test_htest_df_columns(self, htest_report):
        df = htest_report.htest_df()
        assert list(df.columns) == ["x", "y", "gas", "p"]

    def test_pval_df_and_gas_df(self, htest_report):
        assert list(htest_report.pval_df().columns) == ["p"]

        gas = htest_report.gas_df()
        assert list(gas.columns) == ["geneset_A"]

    def test_locations_df(self, htest_report):
        df = htest_report.locations_df()
        assert list(df.columns) == ["x", "y"]

    def test_plot_pval_spatial_map_returns_figure(self, htest_report):
        fig = htest_report.plot_pval_spatial_map()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_pval_spatial_map_with_user_axes(self, htest_report):
        _, ax = plt.subplots()
        fig = htest_report.plot_pval_spatial_map(ax=ax)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_gas_spatial_map_returns_figure(self, htest_report):
        fig = htest_report.plot_gas_spatial_map()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_gas_spatial_map_with_user_axes(self, htest_report):
        _, ax = plt.subplots()
        fig = htest_report.plot_gas_spatial_map(ax=ax)
        assert isinstance(fig, Figure)
        plt.close("all")
