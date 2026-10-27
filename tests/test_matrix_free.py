"""Numerical and memory regressions for the matrix-free computation path."""

import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import linalg as splinalg

from gesso import GESSO
from gesso._src.computation import (
    _eigsh_largest_robust,
    _eigsh_smallest_robust,
    align_gene_contribution_sign,
    bulk_normalize,
    bulk_standard_scale,
    gLPCA_sparse,
    laplacian_spectral_scale,
    standardized_expression,
)

model_module = importlib.import_module("gesso._src.gesso")


def chain_laplacian(n):
    adjacency = sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1])
    return (sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()


@pytest.mark.parametrize("beta", [0, 0.33, 0.9])
@pytest.mark.parametrize("representation", ["dense", "sparse", "implicit"])
def test_scores_match_independent_dense_eigendecomposition(beta, representation):
    rng = np.random.default_rng(31)
    counts = rng.poisson(0.7, size=(8, 35)).astype(float)
    X = bulk_standard_scale(counts)
    L = chain_laplacian(X.shape[1])
    gram = X.T @ X
    lam = np.linalg.eigvalsh(gram)[-1]
    xi = np.linalg.eigvalsh(L.toarray())[-1]
    objective = (1 - beta) * (np.eye(35) - gram / lam) + beta * (
        L.toarray() / xi + np.eye(35) / 35
    )
    _, vectors = np.linalg.eigh(objective)
    v = vectors[:, 0]
    u = X @ v
    scale = np.linalg.norm(u)
    expected_u, expected_v = align_gene_contribution_sign(
        u / scale, v * scale, X, method="sign_overall_expression_proxy"
    )
    inputs = {
        "dense": X,
        "sparse": sparse.csr_matrix(X),
        "implicit": standardized_expression(sparse.csr_matrix(counts)),
    }
    actual_u, actual_v, _, _ = gLPCA_sparse(
        inputs[representation],
        L,
        "example",
        list(range(8)),
        beta=beta,
        verbose=False,
        laplacian_eigenvalue=xi,
    )
    np.testing.assert_allclose(actual_u, expected_u, atol=1e-9)
    np.testing.assert_allclose(actual_v, expected_v, atol=1e-9)
    q = actual_v / np.linalg.norm(actual_v)
    # Use an independent contraction for the residual as well as the oracle.
    action = np.einsum("ij,j->i", objective, q)
    eigenvalue = np.sum(q * action)
    assert np.linalg.norm(action - eigenvalue * q) < 1e-9


def test_implicit_centering_forward_transpose_and_constant_genes():
    counts = np.array(
        [[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0], [0.0, 1.0, 3.0, 0.0]]
    )
    expected = bulk_standard_scale(counts)
    op = standardized_expression(sparse.csr_matrix(counts))
    rng = np.random.default_rng(9)
    v = rng.normal(size=4)
    u = rng.normal(size=3)
    np.testing.assert_allclose(op @ v, expected @ v, atol=1e-12)
    np.testing.assert_allclose(op.T @ u, expected.T @ u, atol=1e-12)
    np.testing.assert_allclose(op @ np.eye(4), expected, atol=1e-12)


def test_existing_knn_objective_matches_explicit_legacy_calculation(gesso_model):
    """Preserve the existing graph semantics; symmetry is a separate issue."""
    X = bulk_standard_scale(gesso_model._expression_df.iloc[:12].to_numpy())
    L = gesso_model._laplacian
    n = X.shape[1]
    rng = np.random.default_rng(42)
    gram = sparse.csr_matrix(X).T @ sparse.csr_matrix(X)
    lam = splinalg.eigsh(
        gram, k=1, which="LA", return_eigenvectors=False, v0=rng.normal(size=n)
    )[0]
    xi = splinalg.eigsh(
        L, k=1, which="LA", return_eigenvectors=False, v0=rng.normal(size=n)
    )[0]
    ident = sparse.eye(n)
    objective = 0.67 * (ident - gram / lam) + 0.33 * (L / xi + ident / n)
    _, q = splinalg.eigsh(objective, k=1, which="SA", v0=rng.normal(size=n))
    u = X @ q[:, 0]
    norm = np.linalg.norm(u)
    old_u, old_v = align_gene_contribution_sign(
        u / norm, q[:, 0] * norm, X, method="sign_overall_expression_proxy"
    )
    new_u, new_v, _, _ = gLPCA_sparse(
        X, L, "g", list(range(12)), beta=0.33, verbose=False
    )
    np.testing.assert_allclose(new_u, old_u, atol=1e-9)
    np.testing.assert_allclose(new_v, old_v, atol=1e-9)


@pytest.mark.parametrize("log1p", [False, True])
@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_sparse_normalization_matches_dense_and_preserves_input(log1p, strategy):
    counts = np.array([[0.0, 0.0, 2.0, 4.0], [0.0, 3.0, 0.0, 8.0]])
    original = counts.copy()
    csr = sparse.csr_matrix(counts)
    result = bulk_normalize(csr, log1p=log1p, rescale_strategy=strategy)
    dense = bulk_normalize(counts, log1p=log1p, rescale_strategy=strategy)
    assert sparse.isspmatrix_csr(result)
    assert result.nnz == csr.nnz
    np.testing.assert_allclose(result.toarray(), dense, atol=1e-14)
    np.testing.assert_array_equal(csr.toarray(), original)
    np.testing.assert_array_equal(counts, original)
    assert np.isfinite(dense).all()
    np.testing.assert_array_equal(dense[:, 0], 0)


@pytest.mark.parametrize("helper", [_eigsh_largest_robust, _eigsh_smallest_robust])
def test_small_operator_fallback(helper, monkeypatch):
    op = splinalg.aslinearoperator(sparse.diags([2.0, 4.0, 6.0]))

    def fail(*args, **kwargs):
        raise splinalg.ArpackNoConvergence("forced", np.array([]), np.empty((3, 0)))

    monkeypatch.setattr(splinalg, "eigsh", fail)
    vals, _ = helper(op)
    expected = 6 if helper is _eigsh_largest_robust else 2
    np.testing.assert_allclose(vals, [expected])


@pytest.mark.parametrize("helper", [_eigsh_largest_robust, _eigsh_smallest_robust])
def test_large_operator_failure_never_materializes_dense(helper, monkeypatch):
    def forbidden_matvec(v):
        pytest.fail("Fallback attempted to materialize the operator")

    op = splinalg.LinearOperator((10000, 10000), matvec=forbidden_matvec, dtype=float)

    def fail(*args, **kwargs):
        raise splinalg.ArpackNoConvergence("forced", np.array([]), np.empty((10000, 0)))

    monkeypatch.setattr(splinalg, "eigsh", fail)
    with pytest.raises(RuntimeError, match="refusing a dense fallback"):
        helper(op)


def test_solver_never_forms_sparse_gram_or_dense_spot_matrix(monkeypatch):
    n = 600
    rng = np.random.default_rng(11)
    counts = sparse.csr_matrix(rng.poisson(0.1, size=(12, n)).astype(float))
    L = chain_laplacian(n)
    original_matmul = sparse.csr_matrix.__matmul__

    def checked_matmul(self, other):
        assert not sparse.issparse(other), "An explicit Gram matrix was formed"
        return original_matmul(self, other)

    def forbidden(*args, **kwargs):
        pytest.fail("Sparse data was densified")

    monkeypatch.setattr(sparse.csr_matrix, "__matmul__", checked_matmul)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbidden)
    _, v, _, _ = gLPCA_sparse(
        standardized_expression(counts),
        L,
        "example",
        list(range(12)),
        beta=0.33,
        verbose=False,
    )
    assert np.isfinite(v).all()


def test_cached_graph_scale_preserves_initial_vector_sequence():
    rng = np.random.default_rng(8)
    X = bulk_standard_scale(rng.normal(size=(8, 40)))
    L = chain_laplacian(40)
    options = dict(
        X=X,
        L=L,
        geneset_name="g",
        genes_in_geneset=list(range(8)),
        beta=0.33,
        verbose=False,
    )
    uncached = gLPCA_sparse(**options)
    cached = gLPCA_sparse(**options, laplacian_eigenvalue=laplacian_spectral_scale(L))
    np.testing.assert_array_equal(uncached[0], cached[0])
    np.testing.assert_array_equal(uncached[1], cached[1])


def test_constant_expression_has_clear_error():
    with pytest.raises(ValueError, match="no nonzero variation"):
        gLPCA_sparse(
            np.zeros((3, 20)), chain_laplacian(20), "g", [1, 2, 3], verbose=False
        )


@pytest.mark.parametrize("normalization", ["none", "normalize", "normalize-log1p"])
@pytest.mark.parametrize("method", ["cpu", "lowres"])
def test_sparse_model_matches_dense_with_alignment(
    expression_df, locations_df, genesets_df, normalization, method, monkeypatch
):
    # Exercise gene/spot alignment and filtering rather than already ordered inputs.
    counts = expression_df.iloc[::-1, ::-1].copy()
    counts.iloc[0] = 0
    locations = locations_df.iloc[2:]
    sparse_counts = counts.astype(pd.SparseDtype("float64", 0))
    original = sparse_counts.copy()
    kwargs = dict(k=4, normalize_counts_method=normalization, verbose=False)
    dense_model = GESSO(counts, locations, genesets_df, **kwargs)
    options = dict(
        n_jobs=1, compute_method=method, n_partitions=3, partition_method="random"
    )
    expected = dense_model.compute_gas(**options)

    def forbidden(*args, **kwargs):
        pytest.fail("Sparse expression data was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbidden)
    sparse_model = GESSO(sparse_counts, locations, genesets_df, **kwargs)
    assert all(
        isinstance(t, pd.SparseDtype) for t in sparse_model._expression_df.dtypes
    )
    actual = sparse_model.compute_gas(**options)
    pd.testing.assert_frame_equal(
        actual.gas_df(), expected.gas_df(), atol=1e-8, rtol=1e-8
    )
    for geneset in genesets_df:
        pd.testing.assert_frame_equal(
            actual.gene_contributions_df(geneset),
            expected.gene_contributions_df(geneset),
            atol=1e-8,
            rtol=1e-8,
        )
    pd.testing.assert_frame_equal(sparse_counts, original)


def test_sparse_nonzero_fill_is_rejected(expression_df, locations_df):
    with pytest.raises(ValueError, match="zero fill"):
        GESSO(expression_df.astype(pd.SparseDtype("float64", np.nan)), locations_df)


def test_graph_scales_cached_across_gene_sets_and_calls(gesso_model):
    with patch.object(
        model_module, "laplacian_spectral_scale", wraps=laplacian_spectral_scale
    ) as scale:
        gesso_model.compute_gas(n_jobs=1)
        gesso_model.compute_gas(n_jobs=1, beta=0.2)
        assert scale.call_count == 1


def test_partition_cache_reused_and_replaced(gesso_model):
    with patch.object(
        gesso_model, "_compute_laplacian_knn", wraps=gesso_model._compute_laplacian_knn
    ) as graph:
        with patch.object(
            model_module, "laplacian_spectral_scale", wraps=laplacian_spectral_scale
        ) as scale:
            options = dict(
                compute_method="lowres",
                partition_method="random",
                n_partitions=3,
                n_jobs=1,
            )
            gesso_model.compute_gas(**options)
            gesso_model.compute_gas(**options, beta=0.2)
            assert graph.call_count == scale.call_count == 3
            gesso_model.compute_gas(**options, partition_seed=7)
            assert graph.call_count == scale.call_count == 6
            options["n_partitions"] = 2
            gesso_model.compute_gas(**options, partition_seed=7)
            assert graph.call_count == scale.call_count == 8
            options["partition_method"] = "stratified_kmeans"
            gesso_model.compute_gas(**options, partition_seed=7)
            assert graph.call_count == scale.call_count == 10


@pytest.mark.parametrize("method", ["cpu", "lowres"])
def test_sparse_model_parallel_matches_serial(
    expression_df, locations_df, genesets_df, method
):
    model = GESSO(
        expression_df.astype(pd.SparseDtype("float64", 0)),
        locations_df,
        genesets_df,
        k=4,
        verbose=False,
    )
    options = dict(compute_method=method, partition_method="random", n_partitions=3)
    serial = model.compute_gas(n_jobs=1, **options).gas_df()
    parallel = model.compute_gas(n_jobs=2, **options).gas_df()
    pd.testing.assert_frame_equal(serial, parallel, atol=1e-10, rtol=1e-10)
