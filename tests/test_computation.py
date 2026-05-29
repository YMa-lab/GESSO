from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from gesso._src.computation import (
    _eigsh_largest_robust,
    _eigsh_smallest_robust,
    align_gene_contribution_sign,
    align_gene_contribution_sign_sparse,
    bulk_normalize,
    bulk_standard_scale,
    check_partition_correctness,
    gLPCA_sparse,
    maybe_flip,
    partition_kmeans_stratified,
    partition_naive,
)


class TestBulkStandardScale:
    def test_axis1_row_zero_mean_unit_std(self):
        x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        out = bulk_standard_scale(x, axis=1)
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(out.std(axis=1), 1.0, atol=1e-12)

    def test_axis0_col_zero_mean_unit_std(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = bulk_standard_scale(x, axis=0)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-12)

    def test_scale_only_preserves_mean_sign(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = bulk_standard_scale(x, axis=1, scale_only=True)

        assert np.all(out > 0)

    def test_zero_variance_row_becomes_zeros(self):

        x = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
        out = bulk_standard_scale(x, axis=1)
        np.testing.assert_array_equal(out[0], np.zeros(3))
        assert np.isfinite(out).all()

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            bulk_standard_scale(np.ones((2, 2)), axis=2)


class TestBulkNormalize:
    def test_median_rescale_preserves_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = bulk_normalize(x, rescale_strategy="median")
        assert out.shape == x.shape

    def test_mean_rescale(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = bulk_normalize(x, rescale_strategy="mean")

        col_sums = x.sum(axis=0)
        np.testing.assert_allclose(out.sum(axis=0), col_sums.mean(), rtol=1e-12)

    def test_log1p_applied(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = bulk_normalize(x, log1p=True)

        assert (out >= 0).all()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="rescale strategy"):
            bulk_normalize(np.ones((2, 2)), rescale_strategy="garbage")


class TestSignAlignment:
    def test_maybe_flip_no_op(self):
        u, v = np.array([1.0, -2.0]), np.array([3.0, 4.0])
        u2, v2 = maybe_flip(u, v, flip=False)
        np.testing.assert_array_equal(u2, u)
        np.testing.assert_array_equal(v2, v)

    def test_maybe_flip_negates(self):
        u, v = np.array([1.0, -2.0]), np.array([3.0, 4.0])
        u2, v2 = maybe_flip(u, v, flip=True)
        np.testing.assert_array_equal(u2, -u)
        np.testing.assert_array_equal(v2, -v)

    def test_none_method_passes_through(self):
        u, v = np.array([-1.0, -2.0]), np.array([-3.0, -4.0])
        u2, v2 = align_gene_contribution_sign(u, v, np.eye(2), method="none")
        np.testing.assert_array_equal(u2, u)
        np.testing.assert_array_equal(v2, v)

    def test_sign_max_abs_flips_when_dominant_negative(self):
        u = np.array([0.1, -5.0, 0.2])
        v = np.array([1.0, 1.0, 1.0])
        u_out, v_out = align_gene_contribution_sign(
            u, v, np.eye(3), method="sign_max_abs"
        )

        assert u_out[1] > 0
        np.testing.assert_array_equal(v_out, -v)

    def test_most_frequent_sign_weights(self):
        u = np.array([1.0, 1.0, -1.0])
        v = np.array([1.0, 1.0, 1.0])
        u_out, _ = align_gene_contribution_sign(
            u, v, np.eye(3), method="most_frequent_sign_weights"
        )
        np.testing.assert_array_equal(u_out, u)

        u = np.array([-1.0, -1.0, 1.0])
        u_out, _ = align_gene_contribution_sign(
            u, v, np.eye(3), method="most_frequent_sign_weights"
        )
        np.testing.assert_array_equal(u_out, -u)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            align_gene_contribution_sign(
                np.ones(2), np.ones(2), np.eye(2), method="bogus"
            )

    def test_sparse_variant_matches_dense_on_dense_input(self):

        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 4))
        u = rng.standard_normal(5)
        v = rng.standard_normal(4)
        u_dense, v_dense = align_gene_contribution_sign(
            u.copy(), v.copy(), X, method="sign_overall_expression_proxy"
        )
        u_sparse, v_sparse = align_gene_contribution_sign_sparse(
            u.copy(),
            v.copy(),
            sparse.csr_matrix(X),
            method="sign_overall_expression_proxy",
        )
        np.testing.assert_allclose(u_dense, u_sparse)
        np.testing.assert_allclose(v_dense, v_sparse)


class TestEigshRobust:
    def test_largest_recovers_known_top_eigenvalue(self):
        A = sparse.diags([1.0, 2.0, 3.0, 10.0]).tocsr()
        vals, _ = _eigsh_largest_robust(A, k=1)
        np.testing.assert_allclose(vals[0], 10.0, rtol=1e-8)

    def test_smallest_recovers_known_bottom_eigenvalue(self):
        A = sparse.diags([1.0, 2.0, 3.0, 10.0]).tocsr()
        vals, _ = _eigsh_smallest_robust(A, k=1)
        np.testing.assert_allclose(vals[0], 1.0, rtol=1e-6)

    def test_largest_returns_eigenvalues_only_when_asked(self):
        A = sparse.diags([1.0, 2.0, 3.0]).tocsr()
        vals = _eigsh_largest_robust(A, k=1, return_eigenvectors=False)

        assert np.asarray(vals).shape[-1] == 1

    def test_dense_fallback_triggers_on_tiny_input(self):

        A = sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 5.0]]))
        vals, _ = _eigsh_largest_robust(A, k=1)
        np.testing.assert_allclose(vals[0], 5.0, rtol=1e-8)


class TestGLPCASparse:
    def _toy_inputs(self, n_genes=10, n_obs=20, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_genes, n_obs))

        diag = 2 * np.ones(n_obs)
        diag[0] = diag[-1] = 1
        L = (
            sparse.diags(diag)
            - sparse.diags(np.ones(n_obs - 1), 1)
            - sparse.diags(np.ones(n_obs - 1), -1)
        )
        L = L.tocsr()
        return X, L

    def test_returns_expected_shapes(self):
        X, L = self._toy_inputs()
        u, v, name, genes = gLPCA_sparse(
            X=X,
            L=L,
            beta=0.3,
            geneset_name="g",
            genes_in_geneset=list("abcdefghij"),
            verbose=False,
        )
        assert u.shape == (X.shape[0],)
        assert v.shape == (X.shape[1],)
        assert name == "g"
        assert len(genes) == X.shape[0]

    def test_u_has_unit_norm(self):
        X, L = self._toy_inputs()
        u, _, _, _ = gLPCA_sparse(
            X=X,
            L=L,
            beta=0.3,
            geneset_name="g",
            genes_in_geneset=list("abcdefghij"),
            verbose=False,
        )
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-8)

    def test_accepts_sparse_X(self):
        X, L = self._toy_inputs()
        u, v, _, _ = gLPCA_sparse(
            X=sparse.csr_matrix(X),
            L=L,
            beta=0.3,
            geneset_name="g",
            genes_in_geneset=list("abcdefghij"),
            verbose=False,
        )
        assert u.shape == (X.shape[0],) and v.shape == (X.shape[1],)

    def test_beta_zero_and_high_both_run(self):
        X, L = self._toy_inputs()
        for beta in (0.0, 0.5, 0.9):
            u, v, _, _ = gLPCA_sparse(
                X=X,
                L=L,
                beta=beta,
                geneset_name="g",
                genes_in_geneset=list("abcdefghij"),
                verbose=False,
            )
            assert np.isfinite(u).all() and np.isfinite(v).all()

    def test_rejects_bad_X_type(self):
        _, L = self._toy_inputs()
        with pytest.raises(ValueError, match="X must be"):
            gLPCA_sparse(
                X="not a matrix",
                L=L,
                beta=0.3,
                geneset_name="g",
                genes_in_geneset=[],
                verbose=False,
            )

    def test_rejects_bad_L_type(self):
        X, _ = self._toy_inputs()
        with pytest.raises(ValueError, match="L must be"):
            gLPCA_sparse(
                X=X,
                L="not a matrix",
                beta=0.3,
                geneset_name="g",
                genes_in_geneset=list("abcdefghij"),
                verbose=False,
            )


class TestPartitioning:
    @pytest.fixture
    def loc_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {"x": rng.standard_normal(40), "y": rng.standard_normal(40)},
            index=[f"s_{i}" for i in range(40)],
        )

    def test_naive_partition_covers_all_indices(self, loc_df):
        parts = partition_naive(loc_df, k=4, seed=1)
        assert len(parts) == 4
        assert check_partition_correctness(parts, loc_df)

    def test_naive_partition_disjoint(self, loc_df):
        parts = partition_naive(loc_df, k=4, seed=1)
        seen: set = set()
        for p in parts:
            assert seen.isdisjoint(set(p))
            seen.update(p)

    def test_naive_partition_is_deterministic(self, loc_df):
        a = partition_naive(loc_df, k=4, seed=7)
        b = partition_naive(loc_df, k=4, seed=7)
        for pa, pb in zip(a, b):
            pd.testing.assert_index_equal(pa, pb)

    def test_naive_rejects_missing_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]})
        with pytest.raises(ValueError, match="'x' and 'y'"):
            partition_naive(df, k=2)

    def test_naive_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            partition_naive([[1, 2], [3, 4]], k=2)

    def test_kmeans_partition_covers_all(self, loc_df):
        parts = partition_kmeans_stratified(loc_df, k=4, seed=1)
        assert len(parts) == 4
        assert check_partition_correctness(parts, loc_df)

    def test_kmeans_partition_disjoint(self, loc_df):
        parts = partition_kmeans_stratified(loc_df, k=4, seed=1)
        seen: set = set()
        for p in parts:
            assert seen.isdisjoint(set(p))
            seen.update(p)

    def test_kmeans_rejects_missing_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="'x' and 'y'"):
            partition_kmeans_stratified(df, k=2)


class TestCheckPartitionCorrectness:
    def test_correct_partition(self):
        df = pd.DataFrame(index=[0, 1, 2, 3])
        parts = [pd.Index([0, 1]), pd.Index([2, 3])]
        assert check_partition_correctness(parts, df) is True

    def test_overlap_detected(self):
        df = pd.DataFrame(index=[0, 1, 2, 3])
        parts = [pd.Index([0, 1, 2]), pd.Index([2, 3])]
        assert check_partition_correctness(parts, df) is False

    def test_missing_index_detected(self):
        df = pd.DataFrame(index=[0, 1, 2, 3])
        parts = [pd.Index([0, 1]), pd.Index([2])]
        assert check_partition_correctness(parts, df) is False
