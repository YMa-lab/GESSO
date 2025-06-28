import numpy as np
import scipy
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from typing import Literal
import time
import pandas as pd
from sklearn.cluster import KMeans
from .console import print_wrapped


def maybe_flip(u: np.ndarray, v: np.ndarray, flip: bool):
    if flip:
        return -u, -v
    return u, v


def align_metagene_sign(
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
    raise ValueError(f"Unknown metagene_sign_assignment_method: {method!r}")


def align_metagene_sign_sparse(
    u_optimal: np.ndarray,
    v_optimal: np.ndarray,
    X: sparse.csr_matrix,
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
        proxy = X.mean(axis=0)
        proxy = np.asarray(proxy).ravel()
        corr = np.corrcoef(v_optimal, proxy)[0, 1]
        return flip_if(np.sign(corr) < 0 or np.isnan(corr))
    raise ValueError(f"Unknown metagene_sign_assignment_method: {method!r}")


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
    x: np.ndarray,
    log1p: bool = False,
    rescale_strategy: Literal["median", "mean"] = "median",
) -> np.ndarray:
    """Bulk normalization of counts per observation down the columns (observations).

    Parameters
    ----------
    x : np.ndarray
        Arbitrary 2D matrix. In the context of SPLAT, this is the gene
        expression matrix. The columns are the observations and the rows are the
        genes.

    log1p : bool
        Default: False. If True, applies log1p transformation.

    Returns
    -------
    np.ndarray
        Count-normalized 2D matrix.
    """
    total_counts = np.sum(x, axis=0)
    x = x / total_counts
    if rescale_strategy == "mean":
        x = x * np.mean(total_counts)
    elif rescale_strategy == "median":
        x = x * np.median(total_counts)
    else:
        raise ValueError("Invalid rescale strategy.")
    if log1p:
        return np.log1p(x)
    return x


def gLPCA_sparse(
    X: np.ndarray | sparse.csr_matrix,
    L: sparse.csr_matrix,
    pathway_name: str,
    genes_in_pathway: list,
    beta: float = 0,
    job_num: int | None = None,
    metagene_sign_assignment_method: Literal[
        "none",
        "sign_max_abs",
        "most_frequent_sign_weights",
        "most_frequent_sign_corrs",
        "sign_overall_expression_proxy",
    ] = "sign_overall_expression_proxy",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """This method implements Theorem 3.1 of the paper Graph-Laplacian PCA:
    Closed-form Solution and Robustness by Bo Jiang, Chris Ding, Bin Luo, and Jin Tang.

    The sparse method is faster but less numerically precise compared to the
    non-sparse method.

    Parameters
    ----------
    X : np.ndarray | sparse.csr_matrix ~ (n_genes, n_obs)
        Gene expression matrix.

    L : sparse.csr_matrix ~ (n_obs, n_obs)
        Graph Laplacian matrix. Must already be sparse.

    pathway_name : str
        Name of the pathway.

    genes_in_pathway : list
        List of genes in the pathway.

    beta : float
        Must be in interval [0, 1].

    job_num : int
        Job number for logging updates.

    metagene_sign_assignment_method : Literal["none", "sign_max_abs", \
        "most_frequent_sign_weights", "most_frequent_sign_corrs"]
        Default: "sign_overall_expression_proxy". As with all PCA/SVD-based methods, SPLAT suffers 
        from a sign ambiguity problem. This parameter sets the heuristics-based 
        method to determine the sign of the metagene weights. The pathway 
        activity scores are modified accordingly.
        Options:
        - "none": None.
        - "sign_max_abs": Multiplies the metagene by `sign(max(abs(metagene)))`.
        - "most_frequent_sign_weights": Multiplies the metagene by the most frequent
            sign of all metagene weights.
        - "most_frequent_sign_corrs": Computes the Pearson correlation between 
            the pathway activity scores and the gene expression for all genes 
            in the pathway. Multiplies the metagene by the most frequent sign of 
            all resulting Pearson correlation coefficients.
        - "sign_overall_expression_proxy": Computes the Pearson correlation
            between the pathway activity scores and the overall expression of
            all genes in the pathway. Multiplies the metagene by the most frequent
            sign of the resulting Pearson correlation coefficients.

    verbose : bool
        Default: True. If False, does not print the message.

    Returns
    -------
    np.ndarray ~ (n_genes)
        1D metagene vector (optimal U vector).

    np.ndarray ~ (n_obs)
        1D pathway activity score vector (optimal V vector).

    str
        Name of the pathway.

    list[str]
        List of genes in the pathway.
    """
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    elif not isinstance(X, sparse.csr_matrix):
        raise ValueError("X must be a numpy array or a sparse csr matrix.")

    if isinstance(L, np.ndarray):
        L = sparse.csr_matrix(L)
    elif not isinstance(L, sparse.csr_matrix):
        raise ValueError("L must be a sparse matrix.")

    start = time.time()
    n = X.shape[1]
    G = X.T @ X
    print_wrapped(
        f"(Job {job_num}: {pathway_name}) " "Computed gram matrix",
        "DEBUG",
        verbose=verbose,
    )

    lmbda = splinalg.eigsh(G, k=1, return_eigenvectors=False, which="LA")[0]
    print_wrapped(
        f"(Job {job_num}: {pathway_name}) " "Computed lmbda", "DEBUG", verbose=verbose
    )

    ident = sparse.eye(n)
    psd_temp = ident - G / lmbda

    xi = splinalg.eigsh(L, k=1, return_eigenvectors=False, which="LA")[0]
    print_wrapped(
        f"(Job {job_num}: {pathway_name}) " "Computed xi", "DEBUG", verbose=verbose
    )

    G_beta = (1 - beta) * psd_temp + beta * (L / xi + ident / n)

    eigenvalue, eigenvector = splinalg.eigsh(G_beta, k=1, which="SA")
    print_wrapped(
        f"(Job {job_num}: {pathway_name}) " "Computed eigenpair",
        "DEBUG",
        verbose=verbose,
    )

    eigenvalue = eigenvalue[0]
    v_optimal = eigenvector.flatten()

    u_optimal = X @ v_optimal

    # scale the metagene vector to have unit norm
    scaling_factor = np.linalg.norm(u_optimal)
    u_optimal = u_optimal / scaling_factor
    v_optimal = v_optimal * scaling_factor

    u_optimal, v_optimal = align_metagene_sign_sparse(
        u_optimal=u_optimal,
        v_optimal=v_optimal,
        X=X,
        method=metagene_sign_assignment_method,
    )

    end = time.time()
    seconds = np.round(end - start, 2)

    if job_num is not None:
        print_wrapped(
            f"(Job {job_num}: {pathway_name}) "
            f"Activity score computation for {pathway_name} completed "
            f"in {seconds} seconds.",
            verbose=verbose,
        )
    else:
        print_wrapped(
            f"Activity score computation for pathway {pathway_name} "
            f"completed in {seconds} seconds.",
            verbose=verbose,
        )

    return u_optimal, v_optimal, pathway_name, genes_in_pathway


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
