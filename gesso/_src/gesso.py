import logging
import os
import warnings
import pandas as pd
import scipy.spatial
import scipy.sparse as sparse
import numpy as np
from joblib import Parallel, delayed
from typing import Literal
import random
from contextlib import nullcontext
from pathlib import Path
from .console import COMPUTE_LOGGER, GENESET_LOGGER, INIT_LOGGER
from .interactive import GeneSetActivityScoresReport, PermutationTestReport
from .permutation import _PermutationAccumulator, _NullScoreStore
from .computation import (
    standardized_expression,
    laplacian_spectral_scale,
    bulk_normalize,
    gLPCA_sparse,
    partition_kmeans_stratified,
    partition_naive,
)

_init_logger = logging.getLogger(INIT_LOGGER)
_compute_logger = logging.getLogger(COMPUTE_LOGGER)


def _expression_values(frame: pd.DataFrame):
    """Extract a numerical matrix without densifying sparse DataFrames."""
    is_sparse = [isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes]
    if any(is_sparse):
        if not all(is_sparse) or any(dtype.fill_value != 0 for dtype in frame.dtypes):
            raise ValueError(
                "Sparse expression DataFrames must use zero fill values in every column."
            )
        return frame.sparse.to_coo().tocsr()
    return frame.to_numpy()


class GESSO:
    """GESSO (Gene sEt activity Score analysis with Spatial lOcation)
    is a model for spatially informed gene set activity analysis.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        genesets_df: pd.DataFrame | None = None,
        k: int = 6,
        normalize_counts_method: Literal[
            "normalize", "normalize-log1p", "none"
        ] = "none",
        verbose: bool = True,
    ):
        """Constructs a GESSO (Gene sEt activity Score analysis with Spatial lOcation)
        model for spatially informed gene set activity analysis. Given spatial
        transcriptomics data and a gene set or pathway, GESSO will return
        a gene set activity score (GAS) for each spatial location (spot).

        Parameters
        ----------
        expression_df : pd.DataFrame ~ (n_spots, n_genes)
            A DataFrame containing n_spots rows and n_genes columns.
            The index will be interpreted as the spot ID.
            The columns will be interpreted as gene names.

            Zero-filled pandas sparse DataFrames are supported without dense
            conversion. For SciPy sparse counts, use
            ``pd.DataFrame.sparse.from_spmatrix(counts, index=spot_ids,
            columns=gene_names)``. Sparse normalization and implicit centering
            preserve sparsity during computation.

        locations_df : pd.DataFrame ~ (n_spots, 2)
            A DataFrame containing n_spots rows and 2 columns.
            The index will be interpreted as the spot ID.
            The index of `locations_df` must match the index of the `expression_df`.
            The columns must be named 'x' and 'y'.
            Each row represents the location (xy coordinates) of that spot.

        genesets_df : pd.DataFrame ~ (n_genes, n_genesets) | None
            Default: None.
            A DataFrame containing n_genes rows and n_genesets columns.
            The index will be interpreted as gene names.
            The columns will be interpreted as geneset names.
            The values must be binary (0 or 1). Entry (i, j) is 1 if gene i is
            in geneset j, and 0 otherwise.
            If None, gene sets can be provided later during GAS computation.

        k : int
            Default: 6. For k-nearest neighbors construction of the
            location graph Laplacian.

        normalize_counts_method : Literal["normalize", "normalize-log1p", "none"]
            Default: "none". How to normalize the counts for each
            spot. If "normalize", first scales the total counts for each
            spot vector (row) to 1, then multiplies each spot vector (row)
            by the median of the total counts for all spot vectors.
            If "normalize-log1p", follows steps for "normalize" but also includes a
            log1p transformation.

        verbose : bool
            Default: True. Per-instance override for emitting log messages from
            this model. When True (default), messages flow through the standard
            ``gesso.*`` loggers; configure output via ``gesso.logging``. When
            False, all messages from this model are suppressed regardless of
            logger configuration.
        """
        # preprocess input data
        if any(isinstance(dtype, pd.SparseDtype) for dtype in expression_df.dtypes):
            # Avoid pandas' generic transpose path for sparse extension arrays.
            self._expression_df = pd.DataFrame.sparse.from_spmatrix(
                _expression_values(expression_df).T,
                index=expression_df.columns,
                columns=expression_df.index,
            )
        else:
            self._expression_df = expression_df.T.copy()
        self._locations_df = locations_df.copy()
        if genesets_df is not None:
            self._genesets_df = genesets_df.copy()
        else:
            self._genesets_df = None

        self._verbose = verbose
        self._original_geneset_sizes: dict[str, int] = {}

        self._force_common_genes()
        self._force_common_cellid()

        self._verify_examples_match()
        self._verify_locations_df()
        self._verify_gene_match()
        self._laplacian = self._compute_laplacian_knn(k=k)
        self._k = k
        self._laplacian_eigenvalue = None
        # Only retain the most recent partition configuration to bound memory.
        self._partition_cache = None

        if normalize_counts_method in ("normalize", "normalize-log1p"):
            values = bulk_normalize(
                _expression_values(self._expression_df),
                log1p=normalize_counts_method == "normalize-log1p",
            )
            constructor = (
                pd.DataFrame.sparse.from_spmatrix
                if sparse.issparse(values)
                else pd.DataFrame
            )
            self._expression_df = constructor(
                values,
                index=self._expression_df.index,
                columns=self._expression_df.columns,
            )
            self._log_init(
                f"Normalized expression data with strategy '{normalize_counts_method}'."
            )
        elif normalize_counts_method != "none":
            raise ValueError("Invalid input for parameter 'normalize_counts'.")
        self._q_cache = None
        self._log_init("Model initialization complete.")

    def _log_init(self, msg: str, level: int = logging.INFO) -> None:
        if self._verbose:
            _init_logger.log(level, msg)

    def _log_compute(self, msg: str, level: int = logging.INFO) -> None:
        if self._verbose:
            _compute_logger.log(level, msg)

    def _get_partition_graphs(self, method, n_partitions, seed):
        """Reuse partitions, graphs and scales independently of the gene sets."""
        key = (method, n_partitions, seed)
        if self._partition_cache is not None and self._partition_cache[0] == key:
            return self._partition_cache[1]
        if method == "random":
            partitioner = partition_naive
        elif method == "stratified_kmeans":
            partitioner = partition_kmeans_stratified
        else:
            raise ValueError(
                f"Invalid input for parameter 'partition_method': {method}."
            )
        # Release the previous configuration before allocating its replacement.
        self._partition_cache = None
        partitions = partitioner(self._locations_df, k=n_partitions, seed=seed)
        graphs = []
        for index in partitions:
            graph = self._compute_laplacian_knn(
                k=self._k, locations_df=self._locations_df.loc[index]
            )
            graphs.append((index, graph, laplacian_spectral_scale(graph)))
        self._partition_cache = (key, graphs)
        return graphs

    def compute_gas(
        self,
        genesets: list[str] | None = None,
        genesets_dict: dict[str, list[str]] | None = None,
        beta: float = 0.33,
        compute_method: Literal["cpu", "lowres"] = "cpu",
        n_jobs: int = -1,
        n_partitions: int | None = None,
        partition_method: Literal["random", "stratified_kmeans"] = "stratified_kmeans",
        partition_seed: int = 42,
        store_gene_contributions: bool = True,
        verbose: bool | None = None,
    ) -> GeneSetActivityScoresReport:
        """
        Parameters
        ----------
        genesets : list[str]
            Default: None.
            A list of gene set names for which the gene set activity scores (GASs)
            should be computed. If None (and genesets_dict is None),
            computes gene set activity scores for all genesets
            provided in the provided genesets DataFrame.

        genesets_dict : dict[str, list[str]] | None
            Default: None.
            A dictionary where the keys are geneset names and the values are lists
            of genes in the geneset. Overrides the genesets parameter.

        beta : float
            Default: 0.33. Must be in the interval [0, 1]. Suggested beta < 0.5.

        compute_method : Literal["cpu", "lowres"]
            The method to use for computation.

        n_jobs : int
            Default: -1. Number of parallel jobs to run. If -1, uses all
            available CPUs.

        n_partitions : int | None
            Default: None. Number of low resolution subsets to use for the lowres
            method. Must be an integer if compute_method is "lowres".
            Ignored if compute_method is "cpu".
            If not specified, uses `n_partitions = int(n_spots / 5000)`.
            If `n_partitions < 2`, uses `n_partitions = 2`.

        partition_method : Literal["random", "stratified_kmeans"]
            Default: "stratified_kmeans". Method to use for partitioning the
            spots into subsets for the low resolution method. Ignored if
            compute_method is "cpu".

        partition_seed : int
            Default: 42. Random seed for reproducibility.

        store_gene_contributions : bool
            Default: True. If True, stores gene contribution values.
            Set to False for memory-intensive tasks that do not require gene contribution values.

        verbose : bool | None
            Default: None. Per-call override. If None, inherits from the model's
            ``verbose`` setting. If False, suppresses all messages (including
            per-geneset worker output) for this call regardless of logger
            configuration. If True, emits messages subject to the configuration
            in ``gesso.logging``.

        Returns
        -------
        GeneSetActivityScoresReport
            A report containing the gene set activity scores DataFrame and
            gene contribution DataFrames (if store_gene_contributions is True).
        """
        if beta < 0 or beta > 1:
            raise ValueError('Parameter "beta" must be in interval [0, 1].')

        if genesets is not None and self._genesets_df is None:
            raise ValueError(
                "Gene sets DataFrame not provided. Cannot compute activity scores."
            )

        if genesets is None and genesets_dict is None:
            if self._genesets_df is None:
                raise ValueError(
                    "Gene sets DataFrame not provided. Cannot compute activity scores."
                )

            genesets = self._genesets_df.columns.to_list()

            if not isinstance(genesets, list):
                raise ValueError('Parameter "genesets" must be a list.')

        elif genesets is None:
            genesets = list(genesets_dict.keys())

        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs < 1:
            n_jobs = 1
        n_jobs = min(len(genesets), n_jobs)

        expression_gene_set = set(self._expression_df.index)
        for geneset_name in genesets:
            if genesets_dict is None:
                n_filtered = int((self._genesets_df[geneset_name] == 1).sum())
                n_original = self._original_geneset_sizes.get(geneset_name, n_filtered)
            else:
                genes = genesets_dict[geneset_name]
                n_original = len(genes)
                n_filtered = sum(1 for g in genes if g in expression_gene_set)
            if n_original > 0 and n_filtered / n_original < 0.05:
                warnings.warn(
                    f"Gene set '{geneset_name}': only {n_filtered}/{n_original} "
                    f"({100 * n_filtered / n_original:.2f}%) genes remain after "
                    f"filtering to genes in the dataset (< 5%). Activity scores "
                    f"for this gene set may be unreliable.",
                    stacklevel=2,
                )

        call_verbose = self._verbose if verbose is None else verbose
        worker_log_level = logging.getLogger(GENESET_LOGGER).getEffectiveLevel()
        # Workers capture the expression data, not the model and its graph caches.
        expression_df = self._expression_df

        # begin computation
        if compute_method == "cpu":
            if call_verbose:
                _compute_logger.info(
                    f"Beginning activity score computation for {len(genesets)} "
                    f"gene sets with {n_jobs} jobs. Method used: {compute_method}."
                )
            gas_df = pd.DataFrame(columns=self._expression_df.columns)
            geneset_to_gene_contributions_df_dict = dict()

            L = self._laplacian
            if self._laplacian_eigenvalue is None:
                self._laplacian_eigenvalue = laplacian_spectral_scale(L)
            graph_scale = self._laplacian_eigenvalue
            method_f = gLPCA_sparse

            def process_geneset(
                geneset: str, genes_in_geneset, job_num: int
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index]:
                # keep only genes present in the expression matrix (preserves order)
                genes_in_geneset = [
                    g for g in genes_in_geneset if g in expression_gene_set
                ]
                X = standardized_expression(
                    _expression_values(expression_df.loc[genes_in_geneset])
                )
                u, v, _, _ = method_f(
                    X=X,
                    L=L,
                    laplacian_eigenvalue=graph_scale,
                    beta=beta,
                    geneset_name=geneset,
                    genes_in_geneset=genes_in_geneset,
                    job_num=job_num,
                    gene_contribution_sign_assignment_method="sign_overall_expression_proxy",
                    verbose=call_verbose,
                    worker_log_level=worker_log_level,
                )
                return geneset, v, u, genes_in_geneset

            if genesets_dict is None:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_geneset)(
                        geneset,
                        self._genesets_df[self._genesets_df[geneset] == 1].index,
                        i + 1,
                    )
                    for i, geneset in enumerate(genesets)
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_geneset)(geneset, genes_in_geneset, i + 1)
                    for i, (geneset, genes_in_geneset) in enumerate(
                        genesets_dict.items()
                    )
                )

            for geneset, v, u, genes_in_geneset in results:
                gas_df.loc[geneset] = v
                if store_gene_contributions:
                    geneset_to_gene_contributions_df_dict[geneset] = pd.DataFrame(
                        u, index=genes_in_geneset, columns=[geneset]
                    )

            return GeneSetActivityScoresReport(
                gas_df=gas_df.transpose(),
                locations_df=self._locations_df,
                geneset_to_gene_contributions_df_dict=geneset_to_gene_contributions_df_dict,
            )

        elif compute_method == "lowres":
            if call_verbose:
                _compute_logger.info(
                    f"Beginning low resolution activity score computation for "
                    f"{len(genesets)} gene sets with {n_jobs} jobs. Method used: "
                    f"{compute_method}."
                )

            if n_partitions is None:
                n_partitions = max(int(len(self._locations_df) / 5000), 2)

            partition_graphs = self._get_partition_graphs(
                partition_method, n_partitions, partition_seed
            )

            method_f = gLPCA_sparse

            def process_geneset(
                geneset: str,
                genes_in_geneset: pd.Index,
                subset_index: pd.Index,
                geneset_num: int,
                subset_num: int,
                job_num: int,
                local_laplacian: sparse.csr_matrix,
                graph_scale: float,
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index, int, int]:
                X = standardized_expression(
                    _expression_values(
                        expression_df.loc[genes_in_geneset, subset_index]
                    )
                )
                u, v, _, _ = method_f(
                    X=X,
                    L=local_laplacian,
                    laplacian_eigenvalue=graph_scale,
                    beta=beta,
                    geneset_name=geneset,
                    genes_in_geneset=genes_in_geneset,
                    job_num=job_num,
                    gene_contribution_sign_assignment_method="sign_overall_expression_proxy",
                    verbose=call_verbose,
                    worker_log_level=worker_log_level,
                )
                return (
                    geneset,
                    v,
                    u,
                    genes_in_geneset,
                    subset_index,
                    geneset_num,
                    subset_num,
                )

            parallel_input_list = []
            job_num = 1
            for geneset_num, geneset in enumerate(genesets):
                if genesets_dict is None:
                    genes_in_geneset = self._genesets_df[
                        self._genesets_df[geneset] == 1
                    ].index
                else:
                    genes_in_geneset = genesets_dict[geneset]
                # keep only genes present in the expression matrix (preserves order)
                genes_in_geneset = [
                    g for g in genes_in_geneset if g in expression_gene_set
                ]
                for subset_num, (subset_index, graph, scale) in enumerate(
                    partition_graphs
                ):
                    parallel_input_list.append(
                        (
                            geneset,
                            genes_in_geneset,
                            subset_index,
                            geneset_num,
                            subset_num,
                            job_num,
                            graph,
                            scale,
                        )
                    )
                    job_num += 1

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_geneset)(*args) for args in parallel_input_list
            )

            geneset_to_reference_gene_idx = {}
            geneset_to_flip_flags = {}
            geneset_to_flip_count = {}
            for result_idx, (geneset, _, u, _, _, _, _) in enumerate(results):
                if geneset not in geneset_to_reference_gene_idx:
                    # first instance of low-res image for geneset
                    geneset_to_reference_gene_idx[geneset] = int(np.argmax(u))
                    geneset_to_flip_flags[geneset] = {result_idx: False}
                    geneset_to_flip_count[geneset] = 0
                else:
                    median_weight = np.median(u)
                    needs_flip = (
                        u[geneset_to_reference_gene_idx[geneset]] < median_weight
                    )
                    geneset_to_flip_flags[geneset][result_idx] = needs_flip
                    geneset_to_flip_count[geneset] += int(needs_flip)

            geneset_to_flip_majority = {}
            for geneset in geneset_to_flip_flags.keys():
                geneset_to_flip_majority[geneset] = (
                    geneset_to_flip_count[geneset]
                    > len(geneset_to_flip_flags[geneset]) / 2
                )

            gas_updates = []
            if store_gene_contributions:
                geneset_to_gene_contributions_list_dict = {g: [] for g in genesets}

            for result_idx, (geneset, v, u, _, subset_index, _, _) in enumerate(
                results
            ):
                flip = geneset_to_flip_flags[geneset][result_idx]
                do_flip = geneset_to_flip_majority[geneset] ^ flip  # flip if needed

                v_final = -v if do_flip else v
                u_final = -u if do_flip else u

                gas_updates.append((geneset, subset_index, v_final))
                if store_gene_contributions:
                    geneset_to_gene_contributions_list_dict[geneset].append(u_final)

            gas_df = pd.DataFrame(
                np.nan, index=genesets, columns=self._expression_df.columns
            )
            # update GAS DataFrame
            for geneset, subset_index, v in gas_updates:
                gas_df.loc[geneset, subset_index] = v

            # average gene contribution values across subsets
            geneset_to_gene_contributions_df_dict = {}
            if store_gene_contributions:
                for (
                    geneset,
                    gene_contributions,
                ) in geneset_to_gene_contributions_list_dict.items():
                    genes_in_geneset = (
                        genesets_dict[geneset]
                        if genesets_dict is not None
                        else self._genesets_df[self._genesets_df[geneset] == 1].index
                    )
                    # match the filtering applied during computation above
                    genes_in_geneset = [
                        g for g in genes_in_geneset if g in expression_gene_set
                    ]
                    gene_contributions_average = np.mean(gene_contributions, axis=0)
                    geneset_to_gene_contributions_df_dict[geneset] = pd.DataFrame(
                        gene_contributions_average,
                        index=genes_in_geneset,
                        columns=[geneset],
                    )

            return GeneSetActivityScoresReport(
                gas_df=gas_df.transpose(),
                locations_df=self._locations_df,
                geneset_to_gene_contributions_df_dict=geneset_to_gene_contributions_df_dict,
            )

        else:
            raise ValueError("Invalid input for parameter 'compute_method'.")

    def htest_elevated_gas(
        self,
        geneset: str | None = None,
        genes_in_geneset: list[str] | None = None,
        beta: float = 0.33,
        n_permutations: int = 500,
        seed: int = 42,
        n_jobs: int = -1,
        *,
        batch_size: int | None = None,
        null_scores_dir: str | os.PathLike | None = None,
    ) -> PermutationTestReport:
        """Conducts a permutation test at each spot to systematically identify
        spots with significantly elevated gene set activity.

        The null hypothesis is that the gene set activity score
        at each spot is not significantly different from the
        activity score of a randomly sampled set of genes
        of the same size as the geneset.

        Parameters
        ----------
        geneset : str | None
            Default: None. Name of the gene set to test. If None, genes_in_geneset must
            be provided.

        genes_in_geneset : list[str] | None
            Default: None. List of genes in the gene set to test. If None, geneset must
            be provided. Overrides geneset if not None.

        beta : float
            Default: 0.33. Must be in the interval [0, 1]. Suggested beta < 0.5.

        n_permutations : int
            Default: 500. Number of random gene sets to sample for the test.

        seed : int
            Default: 42. Random seed for reproducibility.

        n_jobs : int
            Default: -1. Number of parallel jobs to run. If -1, uses all available CPUs.

        batch_size : int | None
            Maximum number of null score vectors collected at once. Defaults to
            twice the worker count. Smaller batches reduce memory use. Sampling
            and p-values are independent of the batch size and worker count.

        null_scores_dir : str | os.PathLike | None
            Optional new output directory for null_scores.npy (permutations by
            spots), observed_scores.npy, locations.json, sampled_genes.jsonl and
            metadata.json. Existing directories are never overwritten. With
            None, no files are written. Failed runs remain marked incomplete.

        Returns
        -------
        PermutationTestReport
            A report containing the gene set activity scores and p-values for each spot.
        """
        if geneset is None and genes_in_geneset is None:
            raise ValueError("Both 'geneset' and 'genes_in_geneset' cannot be None.")
        if (
            isinstance(n_permutations, bool)
            or not isinstance(n_permutations, (int, np.integer))
            or n_permutations < 1
        ):
            raise ValueError("n_permutations must be a positive integer.")
        n_permutations = int(n_permutations)
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        output_dir = (
            None
            if null_scores_dir is None
            else Path(null_scores_dir).expanduser().resolve()
        )
        if output_dir is not None and output_dir.exists():
            raise FileExistsError(
                f"Null-score output directory already exists: {output_dir}"
            )

        all_genes = sorted(self._expression_df.index.to_list())

        if geneset is not None:
            if genes_in_geneset is None:
                if self._genesets_df is None:
                    raise ValueError(
                        "Gene sets DataFrame not provided. Supply genes_in_geneset."
                    )
                genes_in_geneset = self._genesets_df[
                    self._genesets_df[geneset] == 1
                ].index.to_list()
                geneset_name = geneset
            # if both geneset and genes_in_geneset are provided,
            # we use genes_in_geneset, but keep the geneset as geneset name.
            geneset_name = geneset

        else:
            if genes_in_geneset is None:
                raise ValueError(
                    "If 'geneset' is None, 'genes_in_geneset' must be provided."
                )
            geneset_name = "USER_DEFINED"
        genes_in_geneset = list(genes_in_geneset)
        if len(genes_in_geneset) > len(all_genes):
            raise ValueError(
                "The requested gene set is larger than the sampling universe."
            )
        rng = random.Random(seed)
        workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)
        workers = min(n_permutations, workers)
        batch_size = int(batch_size) if batch_size is not None else 2 * workers
        workers = min(workers, batch_size)

        observed = (
            self.compute_gas(
                genesets_dict={geneset_name: genes_in_geneset},
                beta=beta,
                n_jobs=1,
                store_gene_contributions=False,
            )
            .gas_df()[geneset_name]
            .to_numpy()
        )
        accumulator = _PermutationAccumulator(observed)
        expression_df = self._expression_df
        L = self._laplacian
        graph_scale = self._laplacian_eigenvalue
        call_verbose = self._verbose
        worker_log_level = logging.getLogger(GENESET_LOGGER).getEffectiveLevel()

        def score_null(genes, permutation):
            X = standardized_expression(_expression_values(expression_df.loc[genes]))
            _, scores, _, _ = gLPCA_sparse(
                X=X,
                L=L,
                laplacian_eigenvalue=graph_scale,
                beta=beta,
                geneset_name=f"random_geneset_{permutation + 1}",
                genes_in_geneset=genes,
                job_num=permutation + 2,
                verbose=call_verbose,
                worker_log_level=worker_log_level,
            )
            return scores

        store_context = nullcontext(None)
        if output_dir is not None:
            store_context = _NullScoreStore(
                output_dir,
                observed,
                self._locations_df,
                {
                    "geneset": geneset_name,
                    "genes_in_geneset": genes_in_geneset,
                    "gene_universe": all_genes,
                    "beta": float(beta),
                    "seed": seed,
                    "n_permutations": n_permutations,
                },
            )
        with store_context as store:
            # Reuse workers, but bound submitted tasks and returned score vectors.
            with Parallel(
                n_jobs=workers, batch_size=1, pre_dispatch=workers
            ) as parallel:
                for start in range(0, n_permutations, batch_size):
                    batch = [
                        rng.sample(all_genes, len(genes_in_geneset))
                        for _ in range(min(batch_size, n_permutations - start))
                    ]
                    results = parallel(
                        delayed(score_null)(genes, start + i)
                        for i, genes in enumerate(batch)
                    )
                    for genes, scores in zip(batch, results):
                        accumulator.update(scores)
                        if store is not None:
                            store.append(scores, genes)
                    if store is not None:
                        store.checkpoint()
                    del results

        permutation_test_df = self._locations_df[["x", "y"]].copy()
        permutation_test_df["gas"] = observed
        permutation_test_df["p"] = accumulator.p_values()
        return PermutationTestReport(
            geneset=geneset_name,
            permutation_test_df=permutation_test_df,
            null_scores_dir=output_dir,
        )

    def _compute_laplacian_knn(
        self, k: int = 20, locations_df: pd.DataFrame | None = None
    ) -> sparse.csr_matrix:
        """
        Computes the graph laplacian describing topology of
        locations based on k-nearest neighbors.

        Parameters
        ----------
        k : int
            Default: 20. Number of nearest neighbors to connect for each location.

        locations_df : pd.DataFrame | None
            Default: None. DataFrame containing spatial coordinates of spots.
            If None, uses the spatial coordinates provided during initialization.

        Returns
        -------
        sparse.csr_matrix
        """
        if locations_df is not None:
            locations = locations_df[["x", "y"]].values
        else:
            locations = self._locations_df[["x", "y"]].values
        N = locations.shape[0]

        # Use cKDTree for efficient nearest neighbor search
        tree = scipy.spatial.cKDTree(locations)
        _, indices = tree.query(locations, k=k + 1, workers=-1)  # +1 to exclude self

        # Create sparse adjacency matrix
        rows = np.repeat(np.arange(N), k)
        cols = indices[:, 1:].ravel()  # Exclude first column (self)
        data = np.ones(N * k)
        adjacency_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

        # Compute Laplacian
        degrees = adjacency_matrix.sum(axis=1).A1
        laplacian = sparse.diags(degrees) - adjacency_matrix

        self._log_init(
            f"Constructed Laplacian matrix from location data with {k} nearest neighbors.",
            level=logging.DEBUG,
        )

        return laplacian

    def _verify_gene_match(self):
        """Checks that all genes match (i.e., indices of
        self._gene_expression_df and self._genesets_df are equivalent).
        Should be called after preprocessing.
        """
        if self._genesets_df is None:
            return

        if len(self._expression_df) == 0:
            raise ValueError(
                "No genes remain after preprocessing. "
                "Please ensure gene IDs match in gene_expression_df "
                "and genesets_df."
            )

        expression_indices = self._expression_df.index
        geneset_indices = self._genesets_df.index
        if len(expression_indices) != len(geneset_indices):
            raise ValueError(
                "Number of genes in expression_df doesn't match "
                "number of genes in genesets_df"
            )
        if np.array_equal(expression_indices.values, geneset_indices.values):
            return

        def check_match(idx_1, idx_2):
            if idx_1 != idx_2:
                return f"{idx_1} != {idx_2}"
            return None

        results = Parallel(n_jobs=-1)(
            delayed(check_match)(idx1, idx2)
            for idx1, idx2 in zip(expression_indices, geneset_indices)
        )
        mismatches = [result for result in results if result is not None]

        if mismatches:
            raise ValueError(
                "Gene index mismatch following preprocessing: " + ", ".join(mismatches)
            )

    def _verify_examples_match(self):
        """Checks that all examples match (i.e., columns of
        self._gene_expression_df and index of self._locations_df are equivalent).
        Should be called prior to preprocessing.
        """

        def check_match(col_1, idx_2):
            if col_1 != idx_2:
                return f"{col_1} != {idx_2}"
            return None

        columns = self._expression_df.columns
        indices = self._locations_df.index

        if len(columns) != len(indices):
            raise ValueError(
                "Number of columns in expression_df doesn't match number of "
                "indices in locations_df"
            )
        if np.array_equal(columns.values, indices.values):
            return
        results = Parallel(n_jobs=-1)(
            delayed(check_match)(col, idx) for col, idx in zip(columns, indices)
        )
        mismatches = [result for result in results if result is not None]
        if mismatches:
            raise ValueError(
                "Examples column-index mismatch following preprocessing: "
                + ", ".join(mismatches)
            )

    def _verify_genesets(self, genesets: list[str]):
        """Checks that all genesets of interest actually exist in
        self._genesets_df.

        Parameters
        ----------
        genesets : list[str]
        """
        geneset_set = set(self._genesets_df.index)

        # Use numpy for a quick check
        if np.all(np.isin(genesets, list(geneset_set))):
            return

        def check_geneset(geneset):
            if geneset not in geneset_set:
                return geneset
            return None

        results = Parallel(n_jobs=-1)(
            delayed(check_geneset)(geneset) for geneset in genesets
        )
        missing_genesets = [result for result in results if result is not None]

        if missing_genesets:
            raise ValueError(
                "Query gene set(s) not in input geneset df: "
                f"{', '.join(missing_genesets)}"
            )

    def _verify_locations_df(self):
        """
        Checks that the format of locations_df is reasonable.
        Verifies the presence of 'x' and 'y' columns and ensures
        they contain numeric data.
        """
        required_columns = {"x", "y"}
        columns = set(self._locations_df.columns)
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(
                "Missing required columns in locations df: "
                f"{', '.join(missing_columns)}"
            )
        for col in required_columns:
            dtype = self._locations_df[col].dtype
            # pandas 3 uses an extension dtype for strings, which NumPy's
            # issubdtype cannot interpret. Keep rejecting boolean coordinates.
            if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(
                dtype
            ):
                raise ValueError(
                    f"Column '{col}' in locations df must contain numeric data"
                )
        if self._locations_df[list(required_columns)].isnull().any().any():
            raise ValueError("locations df contains NaN values in 'x' or 'y' columns")
        if np.isinf(self._locations_df[list(required_columns)]).any().any():
            raise ValueError(
                "locations df contains infinite values in 'x' or 'y' columns"
            )

    def _force_common_genes(self):
        """
        Finds the common subset of genes. Then, indexes the gene set and
        expression dataframes to only include the common genes.
        """
        if self._genesets_df is None:
            return

        def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            df = df[~df.index.duplicated(keep="first")]
            df.index.name = None
            return df

        self._expression_df = process_dataframe(self._expression_df)
        self._genesets_df = process_dataframe(self._genesets_df)

        # record original geneset sizes prior to common-gene filtering
        self._original_geneset_sizes = (
            (self._genesets_df == 1).sum(axis=0).astype(int).to_dict()
        )

        genes_geneset_set = set(self._genesets_df.index)
        genes_expression_set = set(self._expression_df.index)

        common_genes_set = genes_geneset_set.intersection(genes_expression_set)
        n_genes_removed_geneset = len(genes_geneset_set - common_genes_set)
        n_genes_removed_expression = len(genes_expression_set - common_genes_set)

        if n_genes_removed_geneset > 0:
            self._log_init(
                f"Removed {n_genes_removed_geneset} genes not found in geneset data. "
                f"{len(common_genes_set)} genes remain."
            )
        if n_genes_removed_expression > 0:
            self._log_init(
                f"Removed {n_genes_removed_expression} genes not found in expression "
                f"data. {len(common_genes_set)} genes remain."
            )

        self._log_init(
            f"Identified {len(common_genes_set)} common genes in the gene set "
            "and expression data."
        )
        original_expression_order = self._expression_df.index
        common_genes_list = [
            gene_id
            for gene_id in original_expression_order
            if gene_id in common_genes_set
        ]

        self._genesets_df = self._genesets_df.loc[common_genes_list]
        self._expression_df = self._expression_df.loc[common_genes_list]

    def _force_common_cellid(self):
        """
        Finds the common subset of spot/cell id index between the location and
        expression dataframes. Then, indexes the location and
        expression dataframes to only include the intersection of their indices.
        """
        obs_locations_set = set(self._locations_df.index)
        obs_expression_set = set(self._expression_df.columns)
        common_spots_set = obs_locations_set.intersection(obs_expression_set)
        n_spots_removed_location = len(obs_locations_set - common_spots_set)
        n_spots_removed_expression = len(obs_expression_set - common_spots_set)

        if n_spots_removed_location > 0:
            self._log_init(
                f"Removed {n_spots_removed_location} spots not found in expression "
                f"data. {len(common_spots_set)} spots remain."
            )
        if n_spots_removed_expression > 0:
            self._log_init(
                f"Removed {n_spots_removed_expression} spots not found in location "
                f"data. {len(common_spots_set)} spots remain."
            )

        self._log_init(
            f"Identified {len(common_spots_set)} common spots in the location "
            "and expression data."
        )
        original_loc_order = self._locations_df.index
        common_spots_list = [
            spot_id for spot_id in original_loc_order if spot_id in common_spots_set
        ]
        self._locations_df = self._locations_df.loc[common_spots_list]
        self._expression_df = self._expression_df[common_spots_list]
