import os
import pandas as pd
import scipy.spatial
import scipy.sparse as sparse
import numpy as np
from joblib import Parallel, delayed
from typing import Literal
from .console import print_wrapped
from .interactive import PathwayActivityScoresReport, PermutationTestReport
from .computation import (
    bulk_standard_scale,
    bulk_normalize,
    gLPCA_sparse,
    partition_kmeans_stratified,
    partition_naive,
)


class SPLAT:
    """A SPLAT model for spatially informed gene set expression analysis.
    SPLAT stands for Spatial Pathway Level Analysis Tool.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        pathways_df: pd.DataFrame | None = None,
        k: int = 6,
        normalize_counts_method: Literal[
            "normalize", "normalize-log1p", "none"
        ] = "none",
        verbose: bool = True,
    ):
        """Constructs a SPLAT model.
        SPLAT stands for Spatial Pathway Level Analysis Tool.
        SPLAT computes a pathway activity score (PAS) for each spatial
        location/spot in the provided expression data.

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

        pathways_df : pd.DataFrame ~ (n_genes, n_pathways)
            A DataFrame containing n_genes rows and n_pathways columns.
            The index will be interpreted as gene names.
            The columns will be interpreted as pathway names.
            The values must be binary (0 or 1). Entry (i, j) is 1 if gene i is
            in pathway j, and 0 otherwise.

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
        if pathways_df is not None:
            self._pathways_df = pathways_df.copy()
        else:
            self._pathways_df = None

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

    def compute_pas(
        self,
        pathways: list[str] | None = None,
        pathways_dict: dict[str, list[str]] | None = None,
        beta: float = 0.33,
        compute_method: Literal["cpu", "lowres"] = "cpu",
        n_jobs: int = -1,
        n_partitions: int | None = None,
        partition_method: Literal["random", "stratified_kmeans"] = "stratified_kmeans",
        partition_seed: int = 42,
        store_metagenes: bool = True,
    ) -> PathwayActivityScoresReport:
        """
        Parameters
        ----------
        pathways : list[str]
            Default: None.
            A list of pathway names for which the pathway activity scores should be
            computed. If None (and pathways_dict is None),
            computes pathway activity scores for all pathways
            provided in the provided pathways DataFrame.

        pathways_dict : dict[str, list[str]] | None
            Default: None.
            A dictionary where the keys are pathway names and the values are lists
            of genes in the pathway. Overrides the pathways parameter.

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
        PathwayActivityScoresReport
        """
        if beta < 0 or beta > 1:
            raise ValueError('Parameter "beta" must be in interval [0, 1].')

        if pathways is not None and self._pathways_df is None:
            raise ValueError(
                "Pathways DataFrame not provided. Cannot compute activity scores."
            )

        if pathways is None and pathways_dict is None:
            if self._pathways_df is None:
                raise ValueError(
                    "Pathways DataFrame not provided. Cannot compute activity scores."
                )

            pathways = self._pathways_df.columns.to_list()

            if not isinstance(pathways, list):
                raise ValueError('Parameter "pathways" must be a list.')

        elif pathways is None:
            pathways = list(pathways_dict.keys())

        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs < 1:
            n_jobs = 1
        n_jobs = min(len(pathways), n_jobs)

        # begin computation
        if compute_method == "cpu":
            print_wrapped(
                "Beginning activity score computation "
                f"for {len(pathways)} pathways "
                f"with {n_jobs} jobs. "
                f"Method used: {compute_method}.",
                verbose=self._verbose,
            )
            pas_df = pd.DataFrame(columns=self._expression_df.columns)
            pathway_to_metagene_df_dict = dict()

            L = self._laplacian
            method_f = gLPCA_sparse

            def process_pathway(
                pathway: str, genes_in_pathway: pd.Index, job_num: int
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index]:
                X: np.ndarray = self._expression_df.loc[genes_in_pathway].to_numpy()
                X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=L,
                    beta=beta,
                    pathway_name=pathway,
                    genes_in_pathway=genes_in_pathway,
                    job_num=job_num,
                    metagene_sign_assignment_method="sign_overall_expression_proxy",
                    verbose=self._verbose,
                )
                return pathway, v, u, genes_in_pathway

            if pathways_dict is None:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_pathway)(
                        pathway,
                        self._pathways_df[self._pathways_df[pathway] == 1].index,
                        i + 1,
                    )
                    for i, pathway in enumerate(pathways)
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_pathway)(pathway, genes_in_pathway, i + 1)
                    for i, (pathway, genes_in_pathway) in enumerate(
                        pathways_dict.items()
                    )
                )

            for pathway, v, u, genes_in_pathway in results:
                pas_df.loc[pathway] = v
                if store_metagenes:
                    pathway_to_metagene_df_dict[pathway] = pd.DataFrame(
                        u, index=genes_in_pathway, columns=[pathway]
                    )

            return PathwayActivityScoresReport(
                pas_df=pas_df.transpose(),
                locations_df=self._locations_df,
                pathway_to_metagene_df_dict=pathway_to_metagene_df_dict,
            )

        elif compute_method == "lowres":
            print(
                "Beginning low resolution activity scores computation "
                f"for {len(pathways)} pathways "
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

            def process_pathway(
                pathway: str,
                genes_in_pathway: pd.Index,
                subset_index: pd.Index,
                pathway_num: int,
                subset_num: int,
                job_num: int,
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index, int, int]:
                X: np.ndarray = self._expression_df.loc[
                    genes_in_pathway, subset_index
                ].to_numpy()
                local_laplacian = self._compute_laplacian_knn(
                    k=self._k, locations_df=self._locations_df.loc[subset_index]
                )
                X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=local_laplacian,
                    beta=beta,
                    pathway_name=pathway,
                    genes_in_pathway=genes_in_pathway,
                    job_num=job_num,
                    verbose=self._verbose,
                )
                return (
                    pathway,
                    v,
                    u,
                    genes_in_pathway,
                    subset_index,
                    pathway_num,
                    subset_num,
                )

            print_wrapped(
                "Beginning low resolution activity score computation "
                f"for {len(pathways)} pathways "
                f"with {n_jobs} jobs.",
                verbose=self._verbose,
            )

            parallel_input_list = []
            job_num = 1
            for pathway_num, pathway in enumerate(pathways):
                if pathways_dict is None:
                    genes_in_pathway = self._pathways_df[
                        self._pathways_df[pathway] == 1
                    ].index
                else:
                    genes_in_pathway = pathways_dict[pathway]
                for subset_num, subset_index in enumerate(partitioned_indices):
                    parallel_input_list.append(
                        (
                            pathway,
                            genes_in_pathway,
                            subset_index,
                            pathway_num,
                            subset_num,
                            job_num,
                        )
                    )
                    job_num += 1

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_pathway)(arg0, arg1, arg2, arg3, arg4, arg5)
                for arg0, arg1, arg2, arg3, arg4, arg5 in parallel_input_list
            )

            pathway_to_reference_gene_idx = {}
            pathway_to_flip_flags = {}
            pathway_to_flip_count = {}
            for result_idx, (pathway, _, u, _, _, _, _) in enumerate(results):
                if pathway not in pathway_to_reference_gene_idx:
                    # first instance of low-res image for pathway
                    pathway_to_reference_gene_idx[pathway] = int(np.argmax(u))
                    pathway_to_flip_flags[pathway] = {result_idx: False}
                    pathway_to_flip_count[pathway] = 0
                else:
                    median_weight = np.median(u)
                    needs_flip = (
                        u[pathway_to_reference_gene_idx[pathway]] < median_weight
                    )
                    pathway_to_flip_flags[pathway][result_idx] = needs_flip
                    pathway_to_flip_count[pathway] += int(needs_flip)

            pathway_to_flip_majority = {}
            for pathway in pathway_to_flip_flags.keys():
                pathway_to_flip_majority[pathway] = (
                    pathway_to_flip_count[pathway]
                    > len(pathway_to_flip_flags[pathway]) / 2
                )

            pas_updates = []
            if store_metagenes:
                pathway_to_metagene_list_dict = {p: [] for p in pathways}

            for result_idx, (pathway, v, u, _, subset_index, _, _) in enumerate(
                results
            ):
                flip = pathway_to_flip_flags[pathway][result_idx]
                do_flip = pathway_to_flip_majority[pathway] ^ flip  # flip if needed

                v_final = -v if do_flip else v
                u_final = -u if do_flip else u

                pas_updates.append((pathway, subset_index, v_final))
                if store_metagenes:
                    pathway_to_metagene_list_dict[pathway].append(u_final)

            pas_df = pd.DataFrame(
                np.nan, index=pathways, columns=self._expression_df.columns
            )
            # update PAS DataFrame
            for pathway, subset_index, v in pas_updates:
                pas_df.loc[pathway, subset_index] = v

            # average metagene values across subsets
            pathway_to_metagene_df_dict = {}
            if store_metagenes:
                for pathway, metagenes in pathway_to_metagene_list_dict.items():
                    genes_in_pathway = (
                        pathways_dict[pathway]
                        if pathways_dict is not None
                        else self._pathways_df[self._pathways_df[pathway] == 1].index
                    )
                    metagene_average = np.mean(metagenes, axis=0)
                    pathway_to_metagene_df_dict[pathway] = pd.DataFrame(
                        metagene_average, index=genes_in_pathway, columns=[pathway]
                    )

            return PathwayActivityScoresReport(
                pas_df=pas_df.transpose(),
                locations_df=self._locations_df,
                pathway_to_metagene_df_dict=pathway_to_metagene_df_dict,
            )

        else:
            raise ValueError("Invalid input for parameter 'compute_method'.")

    def htest_elevated_pas(
        self,
        pathway: str | None = None,
        genes_in_pathway: list[str] | None = None,
        beta: float = 0.33,
        n_permutations: int = 500,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> PermutationTestReport:
        """Conducts a permutation test at each spot to systematically identify
        spots with significantly elevated pathway activity.

        The null hypothesis is that the pathway activity score
        at each spot is not significantly different from the
        activity score of a randomly sampled set of genes
        of the same size as the pathway.

        Parameters
        ----------
        pathway : str | None
            Default: None. Name of the pathway to test. If None, genes_in_pathway must
            be provided.

        genes_in_pathway : list[str] | None
            Default: None. List of genes in the pathway to test. If None, pathway must
            be provided. Overrides pathway if not None.

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
            A report containing the pathway activity scores and p-values for each spot.
        """
        if pathway is None and genes_in_pathway is None:
            raise ValueError("Both 'pathway' and 'genes_in_pathway' cannot be None.")

        all_genes = sorted(self._expression_df.index.to_list())

        if pathway is not None:
            if genes_in_pathway is None:
                genes_in_pathway = self._pathways_df[
                    self._pathways_df[pathway] == 1
                ].index.to_list()
                pathway_name = pathway
            # if both pathway and genes_in_pathway are provided,
            # we use genes_in_pathway, but keep the pathway as pathway name.
            pathway_name = pathway

        else:
            if genes_in_pathway is None:
                raise ValueError(
                    "If 'pathway' is None, 'genes_in_pathway' must be provided."
                )
            pathway_name = "USER_DEFINED"

        pathways_dict = {pathway_name: genes_in_pathway}

        # initialize an rng
        rng = np.random.default_rng(seed)

        null_geneset_names = []
        for i in range(n_permutations):
            null_genes = rng.choice(all_genes, len(genes_in_pathway), replace=False)
            random_pathway_name = f"random_geneset_{i}"
            pathways_dict[random_pathway_name] = null_genes
            null_geneset_names.append(random_pathway_name)

        activity_scores_df = self.compute_pas(
            pathways_dict=pathways_dict, beta=beta, n_jobs=n_jobs
        ).pas_df()

        location_index = self._locations_df.index
        # reindex by location index to ensure alignment
        activity_scores_df = activity_scores_df.loc[location_index]

        p_cap = activity_scores_df[pathway_name].to_numpy()
        p_matrix = activity_scores_df[null_geneset_names].to_numpy().T
        prob_greater = np.sum(p_matrix > p_cap, axis=0) / len(p_matrix)
        p_vals = prob_greater
        permutation_test_df = self._locations_df[["x", "y"]].join(
            pd.DataFrame({"p": p_vals}, index=self._locations_df.index)
        )
        # since we already reindexed activity_scores_df by location_index,
        # we can safely assign the pathway activity scores directly
        permutation_test_df["pas"] = activity_scores_df[pathway_name].to_numpy()
        # reorder columns to match expected output
        permutation_test_df = permutation_test_df[["x", "y", "pas", "p"]]
        return PermutationTestReport(
            pathway=pathway_name,
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
        self._gene_expression_df and self._gene_pathway_df are equivalent).
        Should be called after preprocessing.
        """
        if self._pathways_df is None:
            return

        if len(self._expression_df) == 0:
            raise ValueError(
                "No genes remain after preprocessing. "
                "Please ensure gene IDs match in gene_expression_df "
                "and gene_pathway_df."
            )

        expression_indices = self._expression_df.index
        pathway_indices = self._pathways_df.index
        if len(expression_indices) != len(pathway_indices):
            raise ValueError(
                "Number of genes in expression_df doesn't match "
                "number of genes in pathways_df"
            )
        if np.array_equal(expression_indices.values, pathway_indices.values):
            return

        def check_match(idx_1, idx_2):
            if idx_1 != idx_2:
                return f"{idx_1} != {idx_2}"
            return None

        results = Parallel(n_jobs=-1)(
            delayed(check_match)(idx1, idx2)
            for idx1, idx2 in zip(expression_indices, pathway_indices)
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

    def _verify_pathways(self, pathways: list[str]):
        """Checks that all pathways of interest actually exist in
        self._gene_pathway_df.

        Parameters
        ----------
        pathways : list[str]
        """
        pathway_set = set(self._pathways_df.index)

        # Use numpy for a quick check
        if np.all(np.isin(pathways, list(pathway_set))):
            return

        def check_pathway(pathway):
            if pathway not in pathway_set:
                return pathway
            return None

        results = Parallel(n_jobs=-1)(
            delayed(check_pathway)(pathway) for pathway in pathways
        )
        missing_pathways = [result for result in results if result is not None]

        if missing_pathways:
            raise ValueError(
                "Query pathway(s) not in input pathway df: "
                f"{', '.join(missing_pathways)}"
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
        Finds the common subset of genes. Then, indexes the pathway and
        expression dataframes to only include the common genes.
        """
        if self._pathways_df is None:
            return

        def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            # 1) drop any duplicate index values, keeping the first
            df = df[~df.index.duplicated(keep="first")]
            # 2) remove the index name so reset_index()/to_csv() etc. won’t show it
            df.index.name = None
            return df

        self._expression_df = process_dataframe(self._expression_df)
        self._pathways_df = process_dataframe(self._pathways_df)

        genes_pathway_df = set(self._pathways_df.index)
        genes_expression_df = set(self._expression_df.index)
        common_genes = genes_pathway_df.intersection(genes_expression_df)

        n_genes_removed_pathway = len(genes_pathway_df - common_genes)
        n_genes_removed_expression = len(genes_expression_df - common_genes)

        def print_removal_info(n_removed: int, data_type: str):
            if n_removed > 0:
                print_wrapped(
                    f"Removed {n_removed} genes not found in {data_type} data. "
                    f"{len(common_genes)} genes remain.",
                    verbose=self._verbose,
                )

        print_removal_info(n_genes_removed_pathway, "expression")
        print_removal_info(n_genes_removed_expression, "pathway")

        print_wrapped(
            f"Identified {len(common_genes)} common genes in the pathway "
            "and expression data.",
            verbose=self._verbose,
        )

        common_genes = list(common_genes)
        self._pathways_df = self._pathways_df.loc[common_genes]
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
