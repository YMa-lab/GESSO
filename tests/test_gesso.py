from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gesso import GESSO, GeneSetActivityScoresReport, PermutationTestReport


class TestGESSOInit:
    def test_basic_init_succeeds(self, expression_df, locations_df, genesets_df):
        model = GESSO(
            expression_df=expression_df,
            locations_df=locations_df,
            genesets_df=genesets_df,
            verbose=False,
        )
        assert model is not None

    def test_init_without_genesets_df(self, expression_df, locations_df):

        model = GESSO(
            expression_df=expression_df,
            locations_df=locations_df,
            verbose=False,
        )
        assert model is not None

    def test_missing_xy_columns_raises(self, expression_df, locations_df, genesets_df):
        bad_loc = locations_df.rename(columns={"x": "lon"})
        with pytest.raises(ValueError, match="Missing required columns"):
            GESSO(
                expression_df=expression_df,
                locations_df=bad_loc,
                genesets_df=genesets_df,
                verbose=False,
            )

    def test_nonnumeric_locations_raises(
        self, expression_df, locations_df, genesets_df
    ):
        bad_loc = locations_df.copy()
        bad_loc["x"] = bad_loc["x"].astype(str)
        with pytest.raises(ValueError, match="numeric"):
            GESSO(
                expression_df=expression_df,
                locations_df=bad_loc,
                genesets_df=genesets_df,
                verbose=False,
            )

    def test_nan_locations_raises(self, expression_df, locations_df, genesets_df):
        bad_loc = locations_df.copy()
        bad_loc.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            GESSO(
                expression_df=expression_df,
                locations_df=bad_loc,
                genesets_df=genesets_df,
                verbose=False,
            )

    def test_inf_locations_raises(self, expression_df, locations_df, genesets_df):
        bad_loc = locations_df.copy()
        bad_loc.iloc[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            GESSO(
                expression_df=expression_df,
                locations_df=bad_loc,
                genesets_df=genesets_df,
                verbose=False,
            )

    def test_normalize_method(self, expression_df, locations_df, genesets_df):

        for method in ("normalize", "normalize-log1p"):
            model = GESSO(
                expression_df=expression_df,
                locations_df=locations_df,
                genesets_df=genesets_df,
                normalize_counts_method=method,
                verbose=False,
            )
            assert model is not None

    def test_invalid_normalize_method_raises(
        self, expression_df, locations_df, genesets_df
    ):
        with pytest.raises(ValueError, match="normalize_counts"):
            GESSO(
                expression_df=expression_df,
                locations_df=locations_df,
                genesets_df=genesets_df,
                normalize_counts_method="bogus",
                verbose=False,
            )

    def test_common_gene_intersection(self, expression_df, locations_df):

        gs = pd.DataFrame(
            0, index=list(expression_df.columns) + ["phantom_gene"], columns=["gs"]
        )
        gs.loc[expression_df.columns[:5], "gs"] = 1
        gs.loc["phantom_gene", "gs"] = 1
        model = GESSO(
            expression_df=expression_df,
            locations_df=locations_df,
            genesets_df=gs,
            verbose=False,
        )

        assert "phantom_gene" not in model._genesets_df.index

    def test_common_spot_intersection(self, expression_df, locations_df, genesets_df):

        trimmed = locations_df.iloc[:-5]
        model = GESSO(
            expression_df=expression_df,
            locations_df=trimmed,
            genesets_df=genesets_df,
            verbose=False,
        )
        assert model._expression_df.shape[1] == len(trimmed)


class TestComputeGAS:
    def test_compute_gas_with_genesets_dict(
        self, gesso_model_no_genesets, genesets_dict
    ):
        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            n_jobs=1,
        )
        assert isinstance(report, GeneSetActivityScoresReport)
        gas = report.gas_df()
        assert gas.shape == (60, 2)
        assert set(gas.columns) == set(genesets_dict.keys())
        assert np.isfinite(gas.to_numpy()).all()

    def test_compute_gas_from_genesets_df(self, gesso_model, genesets_dict):
        report = gesso_model.compute_gas(n_jobs=1)
        gas = report.gas_df()
        assert gas.shape == (60, len(genesets_dict))

    def test_compute_gas_explicit_geneset_list(self, gesso_model):
        report = gesso_model.compute_gas(genesets=["geneset_A"], n_jobs=1)
        assert report.gas_df().shape[1] == 1

    def test_beta_out_of_range_raises(self, gesso_model):
        with pytest.raises(ValueError, match="beta"):
            gesso_model.compute_gas(beta=-0.1, n_jobs=1)
        with pytest.raises(ValueError, match="beta"):
            gesso_model.compute_gas(beta=1.1, n_jobs=1)

    def test_beta_boundary_values_ok(self, gesso_model, genesets_dict):

        for beta in (0.0, 0.5):
            report = gesso_model.compute_gas(
                genesets_dict={"geneset_A": genesets_dict["geneset_A"]},
                beta=beta,
                n_jobs=1,
            )
            assert np.isfinite(report.gas_df().to_numpy()).all()

    def test_invalid_compute_method_raises(self, gesso_model):
        with pytest.raises(ValueError, match="compute_method"):
            gesso_model.compute_gas(compute_method="gpu", n_jobs=1)

    def test_no_genesets_provided_raises(self, gesso_model_no_genesets):
        with pytest.raises(ValueError, match="Gene sets"):
            gesso_model_no_genesets.compute_gas(n_jobs=1)

    def test_genesets_param_without_genesets_df_raises(self, gesso_model_no_genesets):
        with pytest.raises(ValueError, match="Gene sets"):
            gesso_model_no_genesets.compute_gas(genesets=["geneset_A"], n_jobs=1)

    def test_store_gene_contributions_false(
        self, gesso_model_no_genesets, genesets_dict
    ):
        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            store_gene_contributions=False,
            n_jobs=1,
        )
        assert report._geneset_to_gene_contributions_df_dict == {}

    def test_lowres_method_random_partition(
        self, gesso_model_no_genesets, genesets_dict
    ):
        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            compute_method="lowres",
            partition_method="random",
            n_partitions=3,
            n_jobs=1,
        )
        gas = report.gas_df()
        assert gas.shape == (60, 2)
        assert np.isfinite(gas.to_numpy()).all()

    def test_lowres_method_stratified(self, gesso_model_no_genesets, genesets_dict):
        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            compute_method="lowres",
            partition_method="stratified_kmeans",
            n_partitions=3,
            n_jobs=1,
        )
        gas = report.gas_df()
        assert gas.shape == (60, 2)

    def test_lowres_invalid_partition_method_raises(
        self, gesso_model_no_genesets, genesets_dict
    ):
        with pytest.raises(ValueError, match="partition_method"):
            gesso_model_no_genesets.compute_gas(
                genesets_dict=genesets_dict,
                compute_method="lowres",
                partition_method="bogus",
                n_partitions=2,
                n_jobs=1,
            )

    def test_gas_preserves_spot_order(
        self, gesso_model_no_genesets, genesets_dict, locations_df
    ):
        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            n_jobs=1,
        )

        pd.testing.assert_index_equal(report.gas_df().index, locations_df.index)

    def test_gas_recovers_spatial_signal(
        self, gesso_model_no_genesets, genesets_dict, locations_df
    ):

        report = gesso_model_no_genesets.compute_gas(
            genesets_dict=genesets_dict,
            n_jobs=1,
        )
        gas = report.gas_df()
        x = locations_df["x"].to_numpy()
        corr_a = np.corrcoef(gas["geneset_A"].to_numpy(), x)[0, 1]
        corr_b = np.corrcoef(gas["geneset_B"].to_numpy(), x)[0, 1]

        assert abs(corr_a) > 0.5
        assert abs(corr_b) > 0.5
        assert np.sign(corr_a) != np.sign(corr_b)

    def test_low_coverage_geneset_warns(self, gesso_model_no_genesets, gene_names):
        # only 1 of 30 genes is in the dataset -> ~3.3% coverage (< 5%)
        low_coverage = {
            "sparse_geneset": [gene_names[0]] + [f"absent_{i}" for i in range(29)]
        }
        with pytest.warns(UserWarning, match="genes remain after filtering"):
            gesso_model_no_genesets.compute_gas(
                genesets_dict=low_coverage,
                n_jobs=1,
            )


class TestHTest:
    def test_returns_permutation_report(self, gesso_model, genesets_dict):
        report = gesso_model.htest_elevated_gas(
            geneset="geneset_A",
            n_permutations=20,
            n_jobs=1,
        )
        assert isinstance(report, PermutationTestReport)
        df = report.htest_df()
        assert list(df.columns) == ["x", "y", "gas", "p"]
        assert len(df) == 60

        assert ((df["p"] >= 0) & (df["p"] <= 1)).all()

    def test_with_genes_in_geneset_only(self, gesso_model, genesets_dict):
        report = gesso_model.htest_elevated_gas(
            genes_in_geneset=genesets_dict["geneset_A"],
            n_permutations=20,
            n_jobs=1,
        )
        assert isinstance(report, PermutationTestReport)

        assert report._geneset == "USER_DEFINED"

    def test_both_none_raises(self, gesso_model):
        with pytest.raises(ValueError, match="cannot be None"):
            gesso_model.htest_elevated_gas(n_permutations=5, n_jobs=1)

    def test_reproducibility_with_seed(self, gesso_model):
        a = gesso_model.htest_elevated_gas(
            geneset="geneset_A",
            n_permutations=20,
            seed=123,
            n_jobs=1,
        ).htest_df()
        b = gesso_model.htest_elevated_gas(
            geneset="geneset_A",
            n_permutations=20,
            seed=123,
            n_jobs=1,
        ).htest_df()
        pd.testing.assert_frame_equal(a, b)
