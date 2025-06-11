import numpy as np
import scipy
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from typing import Literal
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


def reconstruction_loss(X: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Reconstruction loss of rank one approximation.
    Computes the Frobenius norm of the difference between X and the outer
    product of u and v.

    Parameters
    ----------
    X : np.ndarray ~ (m, n)
        Arbitrary 2D matrix.

    u : np.ndarray ~ (m)
        1D vector.

    v : np.ndarray ~ (n)
        1D vector.

    Returns
    -------
    float
    """
    return np.linalg.norm(X - np.outer(u, v)).item()


def set_zero_by_threshold(x: np.ndarray, threshold: float = 1e-2) -> np.ndarray:
    """Sets all values in x that are less than threshold away from zero to
    zero.

    Parameters
    ----------
    x : np.ndarray

    threshold : float

    Returns
    -------
    np.ndarray
    """
    x = x.copy()
    x[np.abs(x) < threshold] = 0
    return x


def gLPCA(
    X: np.ndarray,
    L: sparse.csr_matrix,
    pathway_name: str,
    genes_in_pathway: list,
    beta: float = 0,
    job_num: int | None = None,
    metagene_sign_assignment_method: Literal[
        "none", "sign_max_abs", "most_frequent_sign_weights", "most_frequent_sign_corrs"
    ] = "sign_max_abs",
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """This method implements Theorem 3.1 of the paper Graph-Laplacian PCA:
    Closed-form Solution and Robustness by Bo Jiang, Chris Ding, Bin Luo, and Jin Tang

    This version uses dense matrix operations for X and G, but keeps L sparse.

    Parameters
    ----------
    X : np.ndarray ~ (n_genes, n_obs)
        Gene expression matrix.

    L : sparse.csr_matrix ~ (n_obs, n_obs)
        Graph Laplacian matrix. Must be sparse.

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
        Default: "sign_max_abs". As with all PCA/SVD-based methods, SPLAT suffers 
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
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")

    if not isinstance(L, sparse.csr_matrix):
        raise ValueError("L must be a sparse csr matrix.")

    start = time.time()
    n = X.shape[1]
    G = X.T @ X
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed gram matrix", "DEBUG")

    lmbda = scipy.linalg.eigh(
        G, subset_by_index=[G.shape[0] - 1, G.shape[0] - 1], eigvals_only=True
    )[0]
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed lmbda", "DEBUG")

    ident = np.eye(n)
    psd_temp = ident - G / lmbda

    xi = splinalg.eigsh(L, k=1, return_eigenvectors=False, which="LA")[0]
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed xi", "DEBUG")

    # Convert L to dense for this operation
    L_dense = L.toarray()
    G_beta = (1 - beta) * psd_temp + beta * (L_dense / xi + ident / n)

    _, eigenvector = scipy.linalg.eigh(G_beta, subset_by_index=[0, 0])
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed eigenpair", "DEBUG")

    v_optimal = eigenvector[:, 0].flatten()

    u_optimal = X @ v_optimal

    # scale the metagenes to have unit norm
    scaling_factor = np.linalg.norm(u_optimal)
    u_optimal = u_optimal / scaling_factor
    v_optimal = v_optimal * scaling_factor

    u_optimal, v_optimal = align_metagene_sign(
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
            f"in {seconds} seconds."
        )
    else:
        print_wrapped(
            f"Activity score computation for pathway {pathway_name} "
            f"completed in {seconds} seconds."
        )

    return u_optimal, v_optimal, pathway_name, genes_in_pathway


def gLPCA_sparse(
    X: np.ndarray | sparse.csr_matrix,
    L: sparse.csr_matrix,
    pathway_name: str,
    genes_in_pathway: list,
    beta: float = 0,
    job_num: int | None = None,
    metagene_sign_assignment_method: Literal[
        "none", "sign_max_abs", "most_frequent_sign_weights", "most_frequent_sign_corrs"
    ] = "sign_max_abs",
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
        Default: "sign_max_abs". As with all PCA/SVD-based methods, SPLAT suffers 
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
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed gram matrix", "DEBUG")

    lmbda = splinalg.eigsh(G, k=1, return_eigenvectors=False, which="LA")[0]
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed lmbda", "DEBUG")

    ident = sparse.eye(n)
    psd_temp = ident - G / lmbda

    xi = splinalg.eigsh(L, k=1, return_eigenvectors=False, which="LA")[0]
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed xi", "DEBUG")

    G_beta = (1 - beta) * psd_temp + beta * (L / xi + ident / n)

    eigenvalue, eigenvector = splinalg.eigsh(G_beta, k=1, which="SA")
    print_wrapped(f"(Job {job_num}: {pathway_name}) " "Computed eigenpair", "DEBUG")

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
            f"in {seconds} seconds."
        )
    else:
        print_wrapped(
            f"Activity score computation for pathway {pathway_name} "
            f"completed in {seconds} seconds."
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


def partition_simulated_annealing(
    df: pd.DataFrame,
    k: int,
    seed: int = 42,
    bins: int | Literal["auto"] = "auto",
    max_iterations: int = int(1e3),
    cooling_rate: float = 0.98,
    batch_size: int | Literal["auto"] = "auto",
    initialization_method: Literal["stratified_kmeans", "random"] = "stratified_kmeans",
    early_stopping: bool = True,
    early_stopping_interval: int = int(1e2),
    verbose_interval: int = 100,
) -> list[pd.Index]:
    """
    Partition spatial data from a DataFrame into k subsets while maintaining
    similar (x, y) distributions, using simulated annealing with batch swapping.

    This version:
    1) Initializes partitions using either stratified k-means or random partitioning.
    2) Removes the size penalty, since each move swaps points and partition sizes remain constant.
    3) Implements batch swap steps to improve efficiency.
    4) Efficiently updates histograms incrementally.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'x' and 'y'.

    k : int
        Number of partitions to create.

    seed : int
        Random seed for reproducibility.

    bins : int
        Number of bins to use for the 2D histogram.
        If "auto", sets number of bins to int(sqrt(len(df) * 0.2))

    max_iterations : int
        Number of iterations (batches) for simulated annealing.

    cooling_rate : float
        Factor by which temperature is multiplied each iteration.

    batch_size : int | "auto"
        Maximum number of swaps to perform in each batch.
        If "auto", set to 5% of len(df) / k.

    initialization_method : Literal["stratified_kmeans", "random"]
        Method to use for initialization.

    early_stopping : bool
        Whether to stop early if no improvement is made.

    verbose_interval : int
        Interval at which to print updates.
        If 0, no updates are printed.

    Returns
    -------
    list of pandas.Index
        A list of length k, where each element is an Index of row labels for that partition.
    """
    df = df.copy()

    if batch_size == "auto":
        # set batch size to 5% of expected partition size
        batch_size = int(len(df) * 0.05 / k)

    if bins == "auto":
        bins = int(np.sqrt(len(df) * 0.2))

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ["x", "y"]):
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    if initialization_method not in ["stratified_kmeans", "random"]:
        raise ValueError(
            "Initialization method must be 'stratified_kmeans' or 'random'."
        )

    # always reset the index to default for speed
    new_idx_to_original_idx = None
    df["SPLAT_orig_idx"] = df.index.copy()
    df.reset_index(drop=True, inplace=True)
    new_idx_to_original_idx = df["SPLAT_orig_idx"].to_dict()

    scaler = StandardScaler()
    df[["X_scaled", "Y_scaled"]] = scaler.fit_transform(df[["x", "y"]])

    if initialization_method == "stratified_kmeans":
        current_partitions_indices = partition_kmeans_stratified(df, k, seed=seed)
    elif initialization_method == "random":
        current_partitions_indices = partition_naive(df, k, seed=seed)
    else:
        raise ValueError("Unsupported initialization method.")
    current_partitions: list[list] = [
        list(partition) for partition in current_partitions_indices
    ]

    X_scaled = df["X_scaled"].values
    Y_scaled = df["Y_scaled"].values
    global_hist, x_edges, y_edges = np.histogram2d(
        X_scaled, Y_scaled, bins=bins, density=False
    )
    # we want a density histogram, representing joint PMF of (X, Y)
    global_bin_prob = global_hist.flatten()
    # force global_bin_prob to sum to 1 (density=True doesn't really work here)
    global_bin_prob = global_bin_prob / global_bin_prob.sum()

    df["x_bin"] = np.digitize(df["X_scaled"], bins=x_edges) - 1
    df["y_bin"] = np.digitize(df["Y_scaled"], bins=y_edges) - 1
    # ensure bin indices are within [0, bins-1]
    df["x_bin"] = df["x_bin"].clip(0, bins - 1)
    df["y_bin"] = df["y_bin"].clip(0, bins - 1)
    df["bin"] = df["x_bin"] * bins + df["y_bin"]  # unique bin identifier

    partition_bin_counts = np.zeros((k, bins * bins), dtype=int)
    for partition_id, partition in enumerate(current_partitions):
        bin_indices = df.loc[partition, "bin"]
        counts = np.bincount(bin_indices, minlength=bins * bins)
        partition_bin_counts[partition_id] = counts

    def calculate_total_distance(partition_bin_counts_local: np.ndarray):
        """Calculate the total squared distance between each partition's bin
        counts and the target counts.

        Parameters
        ----------
        partition_bin_counts_local : np.ndarray
            2D array of shape (k, bins * bins) containing bin counts for each
            of the k partitions.
        """
        target_point_count = len(df) / k
        target_counts = global_bin_prob * (target_point_count)
        diff = partition_bin_counts_local - target_counts
        return np.sum(diff**2)

    current_distance = calculate_total_distance(partition_bin_counts)
    temperature = 1.0

    # track best partitioning
    best_partitions: list[list] = [list(p) for p in current_partitions]
    best_distance = current_distance
    prev_distance = current_distance

    point_bins = df["bin"].to_dict()

    # simulated annealing algorithm
    for iteration in range(max_iterations):
        if verbose_interval > 0 and (iteration + 1) % verbose_interval == 0:
            print_wrapped(
                f"Simulated Annealing: Iteration {iteration + 1} / {max_iterations}; "
                f"Distance {np.round(best_distance, 1)}; "
                f"Temperature {np.round(temperature, 4)}.",
                level="INFO",
            )
        if (iteration + 1) % early_stopping_interval == 0:
            if early_stopping:
                # check if improvement has been made within the last interval
                if best_distance == prev_distance:
                    # no improvement made
                    if verbose_interval > 0:
                        print_wrapped(
                            "Simulated Annealing: No improvement made at iteration "
                            f"{iteration + 1}. Stopping early.",
                            level="INFO",
                        )
                    break
                else:
                    # reset prev_distance for next interval
                    prev_distance = best_distance

        proposed_swaps = []

        # propose batch of swaps
        for _ in range(batch_size):
            # randomly select two distinct partitions
            p1, p2 = np.random.choice(k, 2, replace=False)

            # ensure both partitions have at least one point to swap
            if len(current_partitions[p1]) == 0 or len(current_partitions[p2]) == 0:
                continue

            # randomly select one point from each partition
            idx1 = np.random.choice(current_partitions[p1])
            idx2 = np.random.choice(current_partitions[p2])

            # avoid proposing the same swap multiple times in the batch
            if (p1, idx1, p2, idx2) in proposed_swaps or (
                p2,
                idx2,
                p1,
                idx1,
            ) in proposed_swaps:
                continue

            proposed_swaps.append((p1, idx1, p2, idx2))

        if not proposed_swaps:
            continue  # no valid swaps proposed in this batch

        # apply all proposed swaps tentatively
        swaps_applied = []

        for swap in proposed_swaps:
            p1, idx1, p2, idx2 = swap

            # get bins for both points
            bin1 = point_bins[idx1]
            bin2 = point_bins[idx2]

            if idx1 not in current_partitions[p1] or idx2 not in current_partitions[p2]:
                continue

            # swap the indices in the partition lists
            current_partitions[p1].remove(idx1)
            current_partitions[p1].append(idx2)
            current_partitions[p2].remove(idx2)
            current_partitions[p2].append(idx1)

            # update partition bin counts
            partition_bin_counts[p1][bin1] -= 1
            partition_bin_counts[p1][bin2] += 1
            partition_bin_counts[p2][bin2] -= 1
            partition_bin_counts[p2][bin1] += 1

            # record the swap for potential reversion
            swaps_applied.append(swap)

        # calculate new distance
        new_distance = calculate_total_distance(partition_bin_counts)
        delta = new_distance - current_distance

        # accept or reject the batch
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            # accept the batch
            current_distance = new_distance
            if new_distance < best_distance:
                best_distance = new_distance
                best_partitions = [list(p) for p in current_partitions]
        else:
            # reject the batch: revert all swaps
            for swap in swaps_applied:
                p1, idx1, p2, idx2 = swap

                # get bins for both points
                bin1 = point_bins[idx1]
                bin2 = point_bins[idx2]

                # swap back the indices in the partition lists
                current_partitions[p1].remove(idx2)
                current_partitions[p1].append(idx1)
                current_partitions[p2].remove(idx1)
                current_partitions[p2].append(idx2)

                # revert partition bin counts
                partition_bin_counts[p1][bin1] += 1
                partition_bin_counts[p1][bin2] -= 1
                partition_bin_counts[p2][bin2] += 1
                partition_bin_counts[p2][bin1] -= 1

        # cool down the temperature
        temperature *= cooling_rate

    partition_indices = [
        pd.Index(p).map(new_idx_to_original_idx) for p in best_partitions
    ]

    # reset the index to original
    df.set_index("SPLAT_orig_idx", inplace=True)

    if not check_partition_correctness(partition_indices, df):
        raise ValueError("Partition is incorrect.")

    return partition_indices


def distance_measure(
    p: np.ndarray,
    q: np.ndarray,
    metric: Literal["euclidean", "cosine", "l1"] = "euclidean",
) -> float:
    """Computes the distance measure between two vectors p and q.

    Parameters
    ----------
    p : np.ndarray
        1D vector.

    q : np.ndarray
        1D vector.

    metric : Literal["euclidean", "cosine", "l1"]
        Default: "euclidean". The metric to use.

    Returns
    -------
    float
    """

    if metric == "cosine":
        return 1 - np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
    elif metric == "euclidean":
        return np.linalg.norm(p - q)
    elif metric == "l1":
        return np.linalg.norm(p - q, ord=1)
    else:
        raise ValueError("Invalid metric.")


def min_max_scale(x: np.ndarray) -> np.ndarray:
    """Min-max scaling for a 1D vector.

    Parameters
    ----------
    x : np.ndarray
        1D vector.

    Returns
    -------
    np.ndarray
    """
    if len(x.shape) != 1:
        raise ValueError("Input must be a 1D vector.")
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def find_median(arr):
    """
    Find the median of a 1D numpy array.

    Parameters
    ----------
        arr (np.ndarray): Input 1D array.

    Returns
    -------
        float: Median value.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    mid = n // 2

    if n % 2 == 0:
        # Even number of elements
        median = (sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0
    else:
        # Odd number of elements
        median = sorted_arr[mid]

    return median
