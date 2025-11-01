import os
import pandas as pd
import scipy.spatial
import scipy.sparse as sparse
import numpy as np
from joblib import Parallel, delayed
from typing import Literal
from .console import print_wrapped
from .interactive import GeneSetActivityScoresReport, PermutationTestReport
from .computation import (
    bulk_standard_scale,
    bulk_normalize,
    gLPCA_sparse,
    partition_kmeans_stratified,
    partition_naive,
)


class GESSO:
    """GESSO (Gene sEt activity Score analysis with Spatial lOcation)
    is a model for spatially informed gene set expression analysis.
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
        model for spatially informed gene set expression analysis. Given spatial
        transcriptomics data and a gene set or pathway, GESSO will return
        a gene set activity score (GAS) for each spatial location (spot).

        Parameters
        ----------
        expression_df : pd.DataFrame ~ (n_spots, n_genes)
            A DataFrame containing n_spots rows and n_genes columns.
            The index will be interpreted as the spot ID.
            The columns will be interpreted as gene names.

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
            Default: True. If True, prints progress messages during initialization.
        """
        # preprocess input data
        self._expression_df = expression_df.T.copy()
        self._locations_df = locations_df.copy()
        if genesets_df is not None:
            self._genesets_df = genesets_df.copy()
        else:
            self._genesets_df = None

        self._verbose = verbose

        self._force_common_genes()
        self._force_common_cellid()

        self._verify_examples_match()
        self._verify_locations_df()
        self._verify_gene_match()
        self._laplacian = self._compute_laplacian_knn(k=k)
        self._k = k

        if normalize_counts_method == "normalize":
            self._expression_df = pd.DataFrame(
                bulk_normalize(self._expression_df.to_numpy(), log1p=False),
                index=self._expression_df.index,
                columns=self._expression_df.columns,
            )
            print_wrapped(
                "Normalized expression data with strategy 'normalize'.", verbose=verbose
            )
        elif normalize_counts_method == "normalize-log1p":
            self._expression_df = pd.DataFrame(
                bulk_normalize(self._expression_df.to_numpy(), log1p=True),
                index=self._expression_df.index,
                columns=self._expression_df.columns,
            )
            print_wrapped(
                "Normalized expression data with strategy 'normalize-log1p'.",
                verbose=verbose,
            )
        elif normalize_counts_method != "none":
            raise ValueError("Invalid input for parameter 'normalize_counts'.")
        self._q_cache = None
        print_wrapped("Model initialization complete.", verbose=verbose)

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
        store_metagenes: bool = True,
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

        compute_method : Literal["cpu-sparse", "cpu", "lowres-sparse", "lowres"]
            The method to use for computation.

        n_jobs : int
            Default: 1. Number of parallel jobs to run. If -1, uses half of
            all available CPUs.

        n_partitions : int | None
            Default: None. Number of low resolution subsets to use for the lowres
            method. Must be an integer if compute_method is "lowres-sparse" or
            "lowres". Ignored if compute_method is "cpu-sparse" or "cpu".
            If not specified, uses `n_partitions = int(n_spots / 5000)`.
            If `n_partitions < 2`, uses `n_partitions = 2`.

        partition_method : Literal["random", "stratified_kmeans"]
            Default: "stratified_kmeans". Method to use for partitioning the
            spots into subsets for the low resolution method. Ignored if
            compute_method is "cpu-sparse" or "cpu".

        partition_seed : int
            Default: 42. Random seed for reproducibility.

        store_metagenes : bool
            Default: True. If True, stores metagene values.
            Set to False for memory-intensive tasks that do not require metagene values.

        Returns
        -------
        GeneSetActivityScoresReport
            A report containing the gene set activity scores DataFrame and
            metagene DataFrames (if store_metagenes is True).
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

        # begin computation
        if compute_method == "cpu":
            print_wrapped(
                "Beginning activity score computation "
                f"for {len(genesets)} gene sets "
                f"with {n_jobs} jobs. "
                f"Method used: {compute_method}.",
                verbose=self._verbose,
            )
            gas_df = pd.DataFrame(columns=self._expression_df.columns)
            geneset_to_metagene_df_dict = dict()

            L = self._laplacian
            method_f = gLPCA_sparse

            def process_geneset(
                geneset: str, genes_in_geneset: pd.Index, job_num: int
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index]:
                X: np.ndarray = self._expression_df.loc[genes_in_geneset].to_numpy()
                X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=L,
                    beta=beta,
                    geneset_name=geneset,
                    genes_in_geneset=genes_in_geneset,
                    job_num=job_num,
                    metagene_sign_assignment_method="sign_overall_expression_proxy",
                    verbose=self._verbose,
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
                if store_metagenes:
                    geneset_to_metagene_df_dict[geneset] = pd.DataFrame(
                        u, index=genes_in_geneset, columns=[geneset]
                    )

            return GeneSetActivityScoresReport(
                gas_df=gas_df.transpose(),
                locations_df=self._locations_df,
                geneset_to_metagene_df_dict=geneset_to_metagene_df_dict,
            )

        elif compute_method == "lowres":
            print(
                "Beginning low resolution activity scores computation "
                f"for {len(genesets)} gene sets "
                f"with {n_jobs} jobs. "
                f"Method used: {compute_method}."
            )

            if n_partitions is None:
                n_partitions = max(int(len(self._locations_df) / 5000), 2)

            if partition_method == "random":
                partitioned_indices = partition_naive(
                    df=self._locations_df, k=n_partitions, seed=partition_seed
                )
            elif partition_method == "stratified_kmeans":
                partitioned_indices = partition_kmeans_stratified(
                    df=self._locations_df, k=n_partitions, seed=partition_seed
                )
            else:
                raise ValueError(
                    f"Invalid input for parameter 'partition_method': "
                    f"{partition_method}."
                )

            method_f = gLPCA_sparse

            def process_geneset(
                geneset: str,
                genes_in_geneset: pd.Index,
                subset_index: pd.Index,
                geneset_num: int,
                subset_num: int,
                job_num: int,
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index, int, int]:
                X: np.ndarray = self._expression_df.loc[
                    genes_in_geneset, subset_index
                ].to_numpy()
                local_laplacian = self._compute_laplacian_knn(
                    k=self._k, locations_df=self._locations_df.loc[subset_index]
                )
                X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=local_laplacian,
                    beta=beta,
                    geneset_name=geneset,
                    genes_in_geneset=genes_in_geneset,
                    job_num=job_num,
                    metagene_sign_assignment_method="sign_overall_expression_proxy",
                    verbose=self._verbose,
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

            print_wrapped(
                "Beginning low resolution activity score computation "
                f"for {len(genesets)} gene sets "
                f"with {n_jobs} jobs.",
                verbose=self._verbose,
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
                for subset_num, subset_index in enumerate(partitioned_indices):
                    parallel_input_list.append(
                        (
                            geneset,
                            genes_in_geneset,
                            subset_index,
                            geneset_num,
                            subset_num,
                            job_num,
                        )
                    )
                    job_num += 1

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_geneset)(arg0, arg1, arg2, arg3, arg4, arg5)
                for arg0, arg1, arg2, arg3, arg4, arg5 in parallel_input_list
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
            if store_metagenes:
                geneset_to_metagene_list_dict = {g: [] for g in genesets}

            for result_idx, (geneset, v, u, _, subset_index, _, _) in enumerate(
                results
            ):
                flip = geneset_to_flip_flags[geneset][result_idx]
                do_flip = geneset_to_flip_majority[geneset] ^ flip  # flip if needed

                v_final = -v if do_flip else v
                u_final = -u if do_flip else u

                gas_updates.append((geneset, subset_index, v_final))
                if store_metagenes:
                    geneset_to_metagene_list_dict[geneset].append(u_final)

            gas_df = pd.DataFrame(
                np.nan, index=genesets, columns=self._expression_df.columns
            )
            # update GAS DataFrame
            for geneset, subset_index, v in gas_updates:
                gas_df.loc[geneset, subset_index] = v

            # average metagene values across subsets
            geneset_to_metagene_df_dict = {}
            if store_metagenes:
                for geneset, metagenes in geneset_to_metagene_list_dict.items():
                    genes_in_geneset = (
                        genesets_dict[geneset]
                        if genesets_dict is not None
                        else self._genesets_df[self._genesets_df[geneset] == 1].index
                    )
                    metagene_average = np.mean(metagenes, axis=0)
                    geneset_to_metagene_df_dict[geneset] = pd.DataFrame(
                        metagene_average, index=genes_in_geneset, columns=[geneset]
                    )

            return GeneSetActivityScoresReport(
                gas_df=gas_df.transpose(),
                locations_df=self._locations_df,
                geneset_to_metagene_df_dict=geneset_to_metagene_df_dict,
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

        Returns
        -------
        PermutationTestReport
            A report containing the gene set activity scores and p-values for each spot.
        """
        if geneset is None and genes_in_geneset is None:
            raise ValueError("Both 'geneset' and 'genes_in_geneset' cannot be None.")

        all_genes = sorted(self._expression_df.index.to_list())

        if geneset is not None:
            if genes_in_geneset is None:
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
        genesets_dict = {geneset_name: genes_in_geneset}

        # initialize an rng
        rng = np.random.default_rng(seed)

        null_geneset_names = []
        for i in range(n_permutations):
            null_genes = rng.choice(all_genes, len(genes_in_geneset), replace=False)
            random_geneset_name = f"random_geneset_{i}"
            genesets_dict[random_geneset_name] = null_genes
            null_geneset_names.append(random_geneset_name)

        activity_scores_df = self.compute_gas(
            genesets_dict=genesets_dict, beta=beta, n_jobs=n_jobs
        ).gas_df()

        location_index = self._locations_df.index
        # reindex by location index to ensure alignment
        activity_scores_df = activity_scores_df.loc[location_index]

        p_cap = activity_scores_df[geneset_name].to_numpy()
        p_matrix = activity_scores_df[null_geneset_names].to_numpy().T
        prob_greater = np.sum(p_matrix > p_cap, axis=0) / len(p_matrix)
        p_vals = prob_greater
        permutation_test_df = self._locations_df[["x", "y"]].join(
            pd.DataFrame({"p": p_vals}, index=self._locations_df.index)
        )
        # since we already reindexed activity_scores_df by location_index,
        # we can safely assign the geneset activity scores directly
        permutation_test_df["gas"] = activity_scores_df[geneset_name].to_numpy()
        # reorder columns to match expected output
        permutation_test_df = permutation_test_df[["x", "y", "gas", "p"]]
        return PermutationTestReport(
            geneset=geneset_name,
            permutation_test_df=permutation_test_df,
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

        print_wrapped(
            "Constructed Laplacian matrix from location data "
            f"with {k} nearest neighbors.",
            level="DEBUG",
            verbose=self._verbose,
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
            if not np.issubdtype(self._locations_df[col].dtype, np.number):
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
            # 1) drop any duplicate index values, keeping the first
            df = df[~df.index.duplicated(keep="first")]
            # 2) remove the index name so reset_index()/to_csv() etc. won’t show it
            df.index.name = None
            return df

        self._expression_df = process_dataframe(self._expression_df)
        self._genesets_df = process_dataframe(self._genesets_df)

        genes_geneset_df = set(self._genesets_df.index)
        genes_expression_df = set(self._expression_df.index)
        common_genes = genes_geneset_df.intersection(genes_expression_df)

        n_genes_removed_geneset = len(genes_geneset_df - common_genes)
        n_genes_removed_expression = len(genes_expression_df - common_genes)

        def print_removal_info(n_removed: int, data_type: str):
            if n_removed > 0:
                print_wrapped(
                    f"Removed {n_removed} genes not found in {data_type} data. "
                    f"{len(common_genes)} genes remain.",
                    verbose=self._verbose,
                )

        print_removal_info(n_genes_removed_geneset, "geneset")
        print_removal_info(n_genes_removed_expression, "expression")

        print_wrapped(
            f"Identified {len(common_genes)} common genes in the gene set "
            "and expression data.",
            verbose=self._verbose,
        )

        common_genes = list(common_genes)
        self._genesets_df = self._genesets_df.loc[common_genes]
        self._expression_df = self._expression_df.loc[common_genes]

    def _force_common_cellid(self):
        """
        Finds the common subset of spot/cell id index between the location and
        expression dataframes. Then, indexes the location and
        expression dataframes to only include the intersection of their indices.
        """

        def get_obs_set(df, attr: str) -> set[str]:
            return set(getattr(df, attr))

        obs_locations_df = get_obs_set(self._locations_df, "index")
        obs_expression_df = get_obs_set(self._expression_df, "columns")
        common_spots = obs_locations_df.intersection(obs_expression_df)

        n_spots_removed_location = len(obs_locations_df - common_spots)
        n_spots_removed_expression = len(obs_expression_df - common_spots)

        def print_removal_info(n_removed: int, data_type: str):
            if n_removed > 0:
                print_wrapped(
                    f"Removed {n_removed} spots not found in {data_type} data. "
                    f"{len(common_spots)} spots remain.",
                    verbose=self._verbose,
                )

        print_removal_info(n_spots_removed_location, "expression")
        print_removal_info(n_spots_removed_expression, "location")

        print_wrapped(
            f"Identified {len(common_spots)} common spots in the location "
            "and expression data.",
            verbose=self._verbose,
        )

        common_spots_list: list[str] = list(common_spots)
        self._locations_df = self._locations_df.loc[common_spots_list]
        self._expression_df = self._expression_df[common_spots_list]
