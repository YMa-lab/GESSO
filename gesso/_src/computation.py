import logging
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from typing import Literal
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils.sparsefuncs import mean_variance_axis
from .console import GENESET_LOGGER, ensure_worker_handler

_geneset_logger = logging.getLogger(GENESET_LOGGER)


def maybe_flip(u: np.ndarray, v: np.ndarray, flip: bool):
    if flip:
        return -u, -v
    return u, v


def _bounded_dense_eigh(A):
    """Only materialize tiny fallback problems, never a large spot matrix."""
    if A.shape[0] > 256:
        raise RuntimeError(
            "Eigenvalue iteration failed; refusing a dense fallback for "
            f"a {A.shape[0]}-spot matrix. Try smaller lowres partitions."
        )
    if isinstance(A, splinalg.LinearOperator):
        dense = A @ np.eye(A.shape[0])
    else:
        dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
    return np.linalg.eigh(dense)


def _eigsh_smallest_robust(A, k=1, v0=None):
    """Retry ARPACK with a larger basis and a bounded small-matrix fallback.

    Operators skip shift-invert, which would introduce an iterative linear
    solve inside every eigenvalue iteration.
    """
    if A.shape[0] <= k:
        vals, vecs = _bounded_dense_eigh(A)
        return vals[:k], vecs[:, :k]
    # default SA
    try:
        return splinalg.eigsh(A, k=k, which="SA", v0=v0)
    except (splinalg.ArpackError, TypeError):
        pass
    # SA with bigger lanczos basis
    try:
        return splinalg.eigsh(
            A, k=k, which="SA", v0=v0, ncv=min(A.shape[0], max(2 * k + 1, 50))
        )
    except (splinalg.ArpackError, TypeError):
        pass
    # shift-invert at small non-zero sigma (avoid 0 if matrix is singular)
    if not isinstance(A, splinalg.LinearOperator) and A.shape[0] <= 256:
        try:
            return splinalg.eigsh(A, k=k, sigma=1e-8, which="LM", v0=v0)
        except (splinalg.ArpackError, TypeError, RuntimeError):
            pass
    # dense fallback
    eigvals, eigvecs = _bounded_dense_eigh(A)
    return eigvals[:k], eigvecs[:, :k]


def _eigsh_largest_robust(A, k=1, return_eigenvectors=True, v0=None):
    """Retry the largest eigenpair with a bounded small-matrix fallback."""
    if A.shape[0] <= k:
        vals, vecs = _bounded_dense_eigh(A)
        return (vals[-k:], vecs[:, -k:]) if return_eigenvectors else vals[-k:]
    # default LA
    try:
        return splinalg.eigsh(
            A, k=k, which="LA", v0=v0, return_eigenvectors=return_eigenvectors
        )
    except (splinalg.ArpackError, TypeError):
        pass
    # LA with bigger lanczos basis
    try:
        return splinalg.eigsh(
            A,
            k=k,
            which="LA",
            v0=v0,
            return_eigenvectors=return_eigenvectors,
            ncv=min(A.shape[0], max(2 * k + 1, 50)),
        )
    except (splinalg.ArpackError, TypeError):
        pass
    # dense fallback
    eigvals, eigvecs = _bounded_dense_eigh(A)
    if return_eigenvectors:
        return eigvals[-k:], eigvecs[:, -k:]
    return eigvals[-k:]


def align_gene_contribution_sign(
    u_optimal: np.ndarray,
    v_optimal: np.ndarray,
    X: np.ndarray,
    method: str = "sign_max_abs",
):
    if method == "none":
        return u_optimal, v_optimal

    def flip_if(cond: bool):
        return maybe_flip(u_optimal, v_optimal, cond)

    if method == "sign_max_abs":
        max_abs_idx = np.argmax(np.abs(u_optimal))
        return flip_if(u_optimal[max_abs_idx] < 0)
    if method == "most_frequent_sign_weights":
        pos = np.count_nonzero(u_optimal > 0)
        neg = np.count_nonzero(u_optimal < 0)
        return flip_if(neg > pos)
    if method == "most_frequent_sign_corrs":
        v_centered = v_optimal - v_optimal.mean()
        covs = X @ v_centered
        pos = np.sum(covs > 0)
        neg = np.sum(covs < 0)
        return flip_if(neg > pos)
    if method == "sign_overall_expression_proxy":
        proxy = X.mean(axis=0)
        corr = np.corrcoef(v_optimal, proxy)[0, 1]
        return flip_if((corr < 0) or np.isnan(corr))
    raise ValueError(f"Unknown gene_contribution_sign_assignment_method: {method!r}")


def align_gene_contribution_sign_sparse(
    u_optimal: np.ndarray,
    v_optimal: np.ndarray,
    X: np.ndarray | sparse.csr_matrix | splinalg.LinearOperator,
    method: str = "sign_max_abs",
):
    if method == "none":
        return u_optimal, v_optimal

    def flip_if(cond: bool):
        return maybe_flip(u_optimal, v_optimal, cond)

    if method == "sign_max_abs":
        max_abs_idx = np.argmax(np.abs(u_optimal))
        return flip_if(u_optimal[max_abs_idx] < 0)
    if method == "most_frequent_sign_weights":
        pos = np.count_nonzero(u_optimal > 0)
        neg = np.count_nonzero(u_optimal < 0)
        return flip_if(neg > pos)
    if method == "most_frequent_sign_corrs":
        v_centered = v_optimal - v_optimal.mean()
        covs = X.dot(v_centered)
        covs = np.asarray(covs).ravel()
        pos = np.sum(covs > 0)
        neg = np.sum(covs < 0)
        return flip_if(neg > pos)
    if method == "sign_overall_expression_proxy":
        if isinstance(X, splinalg.LinearOperator):
            proxy = X.rmatvec(np.full(X.shape[0], 1.0 / X.shape[0]))
        else:
            proxy = X.mean(axis=0)
        proxy = np.asarray(proxy).ravel()
        corr = np.corrcoef(v_optimal, proxy)[0, 1]
        return flip_if(np.sign(corr) < 0 or np.isnan(corr))
    raise ValueError(f"Unknown gene_contribution_sign_assignment_method: {method!r}")


def bulk_standard_scale(
    x: np.ndarray, axis: Literal[0, 1] = 1, scale_only: bool = False
) -> np.ndarray:
    """Bulk standard scaling.

    Parameters
    ----------
    x : np.ndarray
        Arbitrary 2D matrix.

    axis : Literal[0, 1]
        Default: 1. The axis along which to operate.
        Standard scales down columns if axis = 0.

    scale_only : bool
        Default: False. If True, does not center the data to have zero mean for each
        feature.

    Returns
    -------
    np.ndarray
        Standard scaled 2D matrix.
    """
    if axis not in [0, 1]:
        raise ValueError('Parameter "axis" must be 0 or 1.')
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        if scale_only:
            result = np.divide(x, std)
        else:
            result = np.divide(x - mean, std)
        result[~np.isfinite(result)] = 0
    return result


def bulk_normalize(
    x: np.ndarray | sparse.csr_matrix,
    log1p: bool = False,
    rescale_strategy: Literal["median", "mean"] = "median",
) -> np.ndarray | sparse.csr_matrix:
    """Bulk normalization of counts per observation down the columns (observations).

    Parameters
    ----------
    x : np.ndarray | sparse.csr_matrix
        Arbitrary 2D matrix. In the context of GESSO, this is the gene
        expression matrix. The columns are the observations and the rows are the
        genes.

    log1p : bool
        Default: False. If True, applies log1p transformation.

    Returns
    -------
    np.ndarray | sparse.csr_matrix
        Count-normalized matrix, preserving sparse storage. Zero-total spots
        remain zero. The input is not modified.
    """
    total_counts = np.asarray(x.sum(axis=0)).ravel()
    if rescale_strategy == "mean":
        target = np.mean(total_counts)
    elif rescale_strategy == "median":
        target = np.median(total_counts)
    else:
        raise ValueError("Invalid rescale strategy.")
    # Empty spots stay zero in both representations.
    nonzero = total_counts != 0
    if sparse.issparse(x):
        result = x.astype(np.float64, copy=True).tocsr()
        result.sum_duplicates()
        factors = np.divide(
            1.0,
            total_counts,
            out=np.zeros_like(total_counts, dtype=float),
            where=nonzero,
        )
        result.data *= factors[result.indices]
        result.data *= target
        if log1p:
            np.log1p(result.data, out=result.data)
        result.eliminate_zeros()
    else:
        result = np.array(x, dtype=np.float64, copy=True)
        np.divide(result, total_counts, out=result, where=nonzero)
        result[:, ~nonzero] = 0
        result *= target
        if log1p:
            np.log1p(result, out=result)
    return result


def standardized_expression(X):
    """Center/scale genes without filling the implicit zeros of sparse input.

    Represent Z = D X - (D mean) 1.T by its forward and transpose products.
    Dense input keeps the existing explicit scaling path.
    """
    if not sparse.issparse(X):
        return bulk_standard_scale(X, axis=1)
    X = sparse.csr_matrix(X, dtype=np.float64)
    means, variances = mean_variance_axis(X, axis=1)
    std = np.sqrt(np.maximum(variances, 0))
    if not np.any(std > 0):
        raise ValueError("Gene set expression has no nonzero variation.")
    inv_std = np.divide(1.0, std, out=np.zeros_like(std), where=std > 0)
    scaled = X.multiply(inv_std[:, None]).tocsr()
    scaled.eliminate_zeros()
    offset = means * inv_std

    def matvec(v):
        v = np.asarray(v).reshape(-1)
        return scaled @ v - offset * v.sum()

    def rmatvec(u):
        u = np.asarray(u).reshape(-1)
        return scaled.T @ u - offset @ u

    return splinalg.LinearOperator(
        X.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
    )


def laplacian_spectral_scale(L):
    """Compute the graph-only scale with the original deterministic start."""
    rng = np.random.default_rng(42)
    rng.standard_normal(L.shape[0])  # The original first draw belongs to X.T X.
    return float(
        _eigsh_largest_robust(
            L, k=1, return_eigenvectors=False, v0=rng.standard_normal(L.shape[0])
        )[0]
    )


def gLPCA_sparse(
    X: np.ndarray | sparse.csr_matrix | splinalg.LinearOperator,
    L: sparse.csr_matrix,
    geneset_name: str,
    genes_in_geneset: list,
    beta: float = 0,
    job_num: int | None = None,
    gene_contribution_sign_assignment_method: Literal[
        "none",
        "sign_max_abs",
        "most_frequent_sign_weights",
        "most_frequent_sign_corrs",
        "sign_overall_expression_proxy",
    ] = "sign_overall_expression_proxy",
    verbose: bool = True,
    worker_log_level: int = logging.INFO,
    laplacian_eigenvalue: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """This method implements Theorem 3.1 of the paper Graph-Laplacian PCA:
    Closed-form Solution and Robustness by Bo Jiang, Chris Ding, Bin Luo, and Jin Tang.

    Applies the spot-by-spot objective as a LinearOperator without constructing
    the Gram matrix. X may also be an implicitly centered/scaled operator.

    Parameters
    ----------
    X : np.ndarray | sparse.csr_matrix | LinearOperator ~ (n_genes, n_obs)
        Gene expression matrix.

    L : sparse.csr_matrix ~ (n_obs, n_obs)
        Graph Laplacian matrix. Must already be sparse.

    geneset_name : str
        Name of the geneset.

    genes_in_geneset : list
        List of genes in the geneset.

    beta : float
        Must be in interval [0, 1].

    job_num : int
        Job number for logging updates.

    gene_contribution_sign_assignment_method : Literal["none", "sign_max_abs", \
        "most_frequent_sign_weights", "most_frequent_sign_corrs"]
        Default: "sign_overall_expression_proxy". As with all PCA/SVD-based methods, GESSO suffers 
        from a sign ambiguity problem. This parameter sets the heuristics-based 
        method to determine the sign of the gene contribution weights. The geneset 
        activity scores are modified accordingly.
        Options:
        - "none": None.
        - "sign_max_abs": Multiplies the gene contribution vector u by `sign(max(abs(u)))`.
        - "most_frequent_sign_weights": Multiplies the gene contribution by the most frequent
            sign of all gene contribution weights.
        - "most_frequent_sign_corrs": Computes the Pearson correlation between 
            the geneset activity scores and the gene expression for all genes 
            in the geneset. Multiplies the gene contribution by the most frequent sign of 
            all resulting Pearson correlation coefficients.
        - "sign_overall_expression_proxy": Computes the Pearson correlation
            between the geneset activity scores and the overall expression of
            all genes in the geneset. Multiplies the gene contribution by the most frequent
            sign of the resulting Pearson correlation coefficients.

    verbose : bool
        Default: True. If False, suppresses all log messages from this call
        regardless of logger configuration. If True, messages flow through the
        ``gesso.compute.geneset`` logger (configure via ``gesso.logging``).

    worker_log_level : int
        Default: ``logging.INFO``. Effective log level inherited from the
        calling process. Workers spawn fresh and don't inherit the parent's
        logger config, so this is replicated here to honor levels set via
        ``gesso.logging.set_level`` / ``silence_per_geneset`` in the parent.

    laplacian_eigenvalue : float | None
        Optional cached largest eigenvalue of L, shared across gene sets.

    Returns
    -------
    np.ndarray ~ (n_genes)
        1D gene contribution vector (optimal U vector).

    np.ndarray ~ (n_obs)
        1D geneset activity score vector (optimal V vector).

    str
        Name of the geneset.

    list[str]
        List of genes in the geneset.
    """
    if verbose:
        ensure_worker_handler(level=worker_log_level)
        _geneset_logger.setLevel(worker_log_level)

    def _log(msg: str, level: int = logging.INFO) -> None:
        if verbose:
            _geneset_logger.log(level, msg)

    if not isinstance(X, (np.ndarray, sparse.csr_matrix, splinalg.LinearOperator)):
        raise ValueError(
            "X must be a numpy array, sparse csr matrix, or LinearOperator."
        )

    if isinstance(L, np.ndarray):
        L = sparse.csr_matrix(L)
    elif not isinstance(L, sparse.csr_matrix):
        raise ValueError("L must be a sparse matrix.")

    start = time.time()
    n = X.shape[1]
    if X.shape[0] == 0 or n < 2:
        raise ValueError("GESSO requires at least one gene and two spots.")
    if (isinstance(X, np.ndarray) and not np.any(X)) or (
        sparse.issparse(X) and X.count_nonzero() == 0
    ):
        raise ValueError("Gene set expression has no nonzero variation.")
    X_op = splinalg.aslinearoperator(X)
    G = X_op.T @ X_op

    # initialize an rng
    rng = np.random.default_rng(42)

    lmbda = _eigsh_largest_robust(
        G, k=1, return_eigenvectors=False, v0=rng.standard_normal(n)
    )[0]
    _log(f"(Job {job_num}: {geneset_name}) Computed lmbda", logging.DEBUG)

    if not np.isfinite(lmbda) or lmbda <= 0:
        raise ValueError("Gene set expression has no nonzero variation.")
    graph_v0 = rng.standard_normal(n)
    xi = laplacian_eigenvalue
    if xi is None:
        xi = _eigsh_largest_robust(L, k=1, return_eigenvectors=False, v0=graph_v0)[0]
    if not np.isfinite(xi) or xi <= 0:
        raise ValueError("Laplacian must have a positive finite spectral scale.")
    _log(f"(Job {job_num}: {geneset_name}) Computed xi", logging.DEBUG)

    identity_scale = 1 - beta + beta / n

    def objective_matvec(v):
        v = np.asarray(v).reshape(-1)
        return (
            identity_scale * v - ((1 - beta) / lmbda) * (G @ v) + (beta / xi) * (L @ v)
        )

    G_beta = splinalg.LinearOperator((n, n), matvec=objective_matvec, dtype=np.float64)

    eigenvalue, eigenvector = _eigsh_smallest_robust(
        G_beta, k=1, v0=rng.standard_normal(n)
    )
    _log(f"(Job {job_num}: {geneset_name}) Computed eigenpair", logging.DEBUG)

    eigenvalue = eigenvalue[0]
    v_optimal = eigenvector.flatten()

    u_optimal = X @ v_optimal

    # scale the gene contribution vector to have unit norm
    scaling_factor = np.linalg.norm(u_optimal)
    if not np.isfinite(scaling_factor) or scaling_factor == 0:
        raise ValueError(
            "The activity component has zero or nonfinite expression loading."
        )
    u_optimal = u_optimal / scaling_factor
    v_optimal = v_optimal * scaling_factor

    u_optimal, v_optimal = align_gene_contribution_sign_sparse(
        u_optimal=u_optimal,
        v_optimal=v_optimal,
        X=X,
        method=gene_contribution_sign_assignment_method,
    )

    end = time.time()
    seconds = np.round(end - start, 2)

    if job_num is not None:
        _log(
            f"(Job {job_num}: {geneset_name}) Activity score computation for "
            f"{geneset_name} completed in {seconds} seconds."
        )
    else:
        _log(
            f"Activity score computation for geneset {geneset_name} completed "
            f"in {seconds} seconds."
        )

    return u_optimal, v_optimal, geneset_name, genes_in_geneset


def check_partition_correctness(partition: list[pd.Index], df: pd.DataFrame) -> bool:
    """
    Check if a partition is correct, i.e., it contains all indices,
    and no index is repeated between partitions.
    """
    all_indices = set(df.index)
    partition_indices = set()
    for part in partition:
        if not partition_indices.isdisjoint(part):
            return False
        partition_indices.update(part)
    return partition_indices == all_indices


def partition_naive(df: pd.DataFrame, k: int, seed: int = 42) -> list[pd.Index]:
    """Returns a naive partition of a spatial DataFrame into k subsets
    (completely at random).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'x' and 'y'.

    k : int
        Number of partitions to create.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of Index
        List of k pandas.Index objects containing indices for each partition.
    """
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ["x", "y"]):
        raise ValueError("DataFrame must contain 'x' and 'y' columns")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(df.index)
    partition_size = len(df) // k
    remainder = len(df) % k

    partitions = []
    start_idx = 0
    for i in range(k):
        end_idx = start_idx + partition_size + (1 if i < remainder else 0)
        partitions.append(pd.Index(shuffled_indices[start_idx:end_idx]))
        start_idx = end_idx

    if not check_partition_correctness(partitions, df):
        raise ValueError("Partition is incorrect.")

    return partitions


def partition_kmeans_stratified(
    df: pd.DataFrame, k: int, seed: int = 42
) -> list[pd.Index]:
    """
    Partition spatial data from a DataFrame into k subsets using a stratified k-means approach.

    Steps:
    1) Run k-means with k clusters.
    2) For each cluster, shuffle its points and distribute them among all k partitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'x' and 'y'.

    k : int
        Number of partitions to create.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of pandas.Index
        A list of length k, where each element is an Index of row labels for that partition.
    """
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ["x", "y"]):
        raise ValueError("DataFrame must contain 'x' and 'y' columns")

    np.random.seed(seed)
    kmeans = KMeans(n_clusters=k, random_state=seed)
    df["cluster_label"] = kmeans.fit_predict(df[["x", "y"]])

    partitions = [[] for _ in range(k)]
    for cluster_id in range(k):
        # all points in this cluster
        cluster_indices = df.index[df["cluster_label"] == cluster_id].tolist()
        np.random.shuffle(cluster_indices)
        size = len(cluster_indices)
        base_chunk = size // k  # minimum number of points per partition
        remainder = size % k  # leftover points we have to distribute one by one
        start = 0
        for partition_id in range(k):
            chunk_size = base_chunk + (1 if partition_id < remainder else 0)
            end = start + chunk_size
            if chunk_size > 0:
                partitions[partition_id].extend(cluster_indices[start:end])
            start = end

    partition_indices = [pd.Index(part) for part in partitions]
    df.drop(columns="cluster_label", inplace=True)

    if not check_partition_correctness(partitions, df):
        raise ValueError("Partition is incorrect.")

    return partition_indices
