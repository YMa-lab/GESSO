import os
import pandas as pd
import scipy.stats
import scipy.spatial
import scipy.sparse as sparse
import numpy as np
from joblib import Parallel, delayed
from typing import Literal
from .console import print_wrapped
from .interactive import SplatReport
from .computation import (
    bulk_standard_scale,
    bulk_normalize,
    gLPCA,
    gLPCA_sparse,
    partition_kmeans_stratified,
    partition_naive,
)


class Splat:
    """A SPLAT model for spatially informed gene set expression analysis.
    SPLAT stands for Spatial Pathway Level Analysis Tool.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        location_df: pd.DataFrame,
        pathways_df: pd.DataFrame | None = None,
        k: int = 6,
        normalize_counts_method: Literal[
            "normalize", "normalize-log1p", "none"
        ] = "none",
    ):
        """Constructs a SPLAT model.
        SPLAT stands for Spatial Pathway Level Analysis Tool.


        Parameters
        ----------
        expression_df : pd.DataFrame ~ (n_genes, n_obs)
            A DataFrame containing n_genes rows and n_obs columns. Each row
            corresponds to a gene and each column corresponds to an observation.
            Entry (i, j) is the raw expression level of gene i in observation j
            (i.e., number of counts).

        location_df : pd.DataFrame ~ (n_obs, 2)
            A DataFrame containing n_obs rows and 2 columns. The columns must
            be named 'x' and 'y'. Each row corresponds to the location
            (x, y coordinates) of an observation.

        pathways_df : pd.DataFrame ~ (n_genes, n_pathways)
            A DataFrame containing n_genes rows and n_pathways columns. Each
            row corresponds to a gene and each column corresponds to a pathway.
            The values must be binary (0 or 1). Entry (i, j) is 1 if gene i is
            in pathway j, and 0 otherwise.

        k : int
            Default: 6. For k-nearest neighbors construction of the
            location graph Laplacian.

        normalize_counts_method : Literal["normalize", "normalize-log1p", "none"]
            Default: "none". How to normalize the counts for each
            observation. If "normalize", first scales the total counts for each
            observation vector (column) to 1,
            then multiplies each observation vector (column)
            by the median of the total counts for all observation vectors.
            If "normalize-log1p", follows steps for "normalize" but also includes a
            log1p transformation.
        """

        # preprocess input data
        self._expression_df = expression_df.copy()
        self._locations_df = location_df.copy()
        if pathways_df is not None:
            self._pathways_df = pathways_df.copy()
        else:
            self._pathways_df = None

        self._force_common_genes()
        self._force_common_observation()

        self._verify_examples_match()
        self._verify_location_df()
        self._verify_gene_match()
        self._laplacian = self._compute_laplacian_knn(k=k)
        self._k = k

        if normalize_counts_method == "normalize":
            self._expression_df = pd.DataFrame(
                bulk_normalize(self._expression_df.to_numpy(), log1p=False),
                index=self._expression_df.index,
                columns=self._expression_df.columns,
            )
            print_wrapped("Normalized expression data with strategy `normalize`.")
        elif normalize_counts_method == "normalize-log1p":
            self._expression_df = pd.DataFrame(
                bulk_normalize(self._expression_df.to_numpy(), log1p=True),
                index=self._expression_df.index,
                columns=self._expression_df.columns,
            )
            print_wrapped("Normalized expression data with strategy `normalize-log1p`.")
        elif normalize_counts_method != "none":
            raise ValueError("Invalid input for parameter `normalize_counts`.")
        self._q_cache = None
        print_wrapped("Model initialization complete.")

    def generate_activity_scores_report(
        self,
        pathways: list[str] | None = None,
        pathways_dict: dict[str, list[str]] | None = None,
        beta: float = 0.25,
        compute_method: Literal[
            "cpu-sparse", "cpu", "lowres-sparse", "lowres"
        ] = "cpu-sparse",
        transform_method: Literal["none", "standardize"] = "standardize",
        n_jobs: int = -1,
        n_partitions: int | None = None,
        partition_method: Literal["random", "stratified_kmeans"] = "stratified_kmeans",
        partition_seed: int = 42,
        metagene_sign_assignment_method: Literal[
            "none",
            "sign_max_abs",
            "most_frequent_sign_weights",
            "most_frequent_sign_corrs",
            "sign_overall_expression_proxy",
        ] = "sign_overall_expression_proxy",
        store_metagenes: bool = True,
    ) -> SplatReport:
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
            Default: 0.25. Must be in the interval [0, 1]. Suggested beta < 0.5.

        compute_method : Literal["cpu-sparse", "cpu", "lowres-sparse", "lowres"]
            The method to use for computation.

        transform_method : Literal["none", "standardize"]
            The method to use for transforming the data matrix.
            If "standardize", centers and scales each feature vector (rows) to
            have mean 0 and std 1. Default: "standardize".

        n_jobs : int
            Default: 1. Number of parallel jobs to run. If -1, uses half of
            all available CPUs.

        n_partitions : int | None
            Default: None. Number of low resolution subsets to use for the lowres
            method. Must be an integer if compute_method is "lowres-sparse" or
            "lowres". Ignored if compute_method is "cpu-sparse" or "cpu".
            If not specified, uses `n_partitions = int(n_obs / 5000)`.
            If `n_partitions < 2`, uses `n_partitions = 2`.

        partition_method : Literal["random", "stratified_kmeans"]
            Default: "stratified_kmeans". Method to use for partitioning the
            observations into subsets for the low resolution method. Ignored if
            compute_method is "cpu-sparse" or "cpu".

        partition_seed : int
            Default: 42. Random seed for reproducibility.

        metagene_sign_assignment_method : Literal["none", "sign_max_abs", \
            "most_frequent_sign_weights", "most_frequent_sign_corrs", \
            "sign_overall_expression_proxy"]
            Default: "sign_overall_expression_proxy". 
            As with all PCA/SVD-based methods, SPLAT suffers 
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
            - "sign_overall_expression_proxy": Builds an “overall-expression proxy” 
                for each spot—e.g., the (optionally |u|-weighted) mean expression of 
                all genes in the pathway—then computes the Pearson correlation 
                between this proxy and the pathway-activity vector v. 
                The metagene (and v) are multiplied by sign(corr), 
                ensuring that positive activity scores correspond to higher aggregate 
                pathway expression in the original matrix X.

        store_metagenes : bool
            Default: True. If True, stores metagene values.
            Set to False for memory-intensive tasks that do not require metagene values.

        Returns
        -------
        SplatReport
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

        if compute_method in ["cpu-sparse", "cpu"]:
            print_wrapped(
                "Beginning activity score computation "
                f"for {len(pathways)} pathways "
                f"with {n_jobs} jobs. "
                f"Method used: {compute_method}."
            )

            pas_df = pd.DataFrame(columns=self._expression_df.columns)
            pathway_to_metagene_df_dict = dict()

            L = self._laplacian

            if compute_method == "cpu-sparse":
                method_f = gLPCA_sparse
            elif compute_method == "cpu":
                method_f = gLPCA

            def process_pathway(
                pathway: str, genes_in_pathway: pd.Index, job_num: int
            ) -> tuple[str, np.ndarray, np.ndarray, pd.Index]:
                X: np.ndarray = self._expression_df.loc[genes_in_pathway].to_numpy()
                if transform_method == "standardize":
                    X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=L,
                    beta=beta,
                    pathway_name=pathway,
                    genes_in_pathway=genes_in_pathway,
                    job_num=job_num,
                    metagene_sign_assignment_method=metagene_sign_assignment_method,
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

            return SplatReport(
                pas_df=pas_df.transpose(),
                location_df=self._locations_df,
                pathway_to_metagene_df_dict=pathway_to_metagene_df_dict,
            )

        elif compute_method in ["lowres-sparse", "lowres"]:
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
                    f"Invalid input for parameter `partition_method`: "
                    f"{partition_method}."
                )

            if compute_method == "lowres-sparse":
                method_f = gLPCA_sparse
            elif compute_method == "lowres":
                method_f = gLPCA

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
                if transform_method == "standardize":
                    X = bulk_standard_scale(X, axis=1)
                u, v, _, _ = method_f(
                    X=X,
                    L=local_laplacian,
                    beta=beta,
                    pathway_name=pathway,
                    genes_in_pathway=genes_in_pathway,
                    job_num=job_num,
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
                f"with {n_jobs} jobs."
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

            return SplatReport(
                pas_df=pas_df.transpose(),
                location_df=self._locations_df,
                pathway_to_metagene_df_dict=pathway_to_metagene_df_dict,
            )

        else:
            raise ValueError("Invalid input for parameter `compute_method`.")

    def test_pathway(
        self,
        pathway: str | None = None,
        genes_in_pathway: list[str] | None = None,
        beta: float = 0.25,
        control_size: int = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> float:
        """Conducts a permutation test to determine if the pathway is spatially
        significantly different from control genes.

        Parameters
        ----------
        pathway : str | None
            Default: None. Name of the pathway to test. If None, genes_in_pathway must
            be provided.

        genes_in_pathway : list[str] | None
            Default: None. List of genes in the pathway to test. If None, pathway must
            be provided. Overrides pathway if not None.

        beta : float
            Default: 0.25. Must be in the interval [0, 1]. Suggested beta < 0.5.

        control_size : int
            Default: 100. Number of control genes to sample for the permutation test.

        seed : int
            Default: 42. Random seed for reproducibility.

        n_jobs : int
            Default: -1. Number of parallel jobs to run. If -1, uses all available CPUs.

        Returns
        -------
        float
            The p-value of the test. The p-value is the proportion of
            control genes that have a higher pathway activity score than the observed
            pathway activity score.
        """
        if pathway is None and genes_in_pathway is None:
            raise ValueError("Both pathway and genes_in_pathway cannot be None.")

        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs < 1:
            n_jobs = 1

        all_genes = self._expression_df.index.to_list()

        if pathway is not None:
            genes_in_pathway = self._pathways_df[
                self._pathways_df[pathway] == 1
            ].index.to_list()
        p_pathway_name = "SPLAT:::test_pathway:::p"
        pathways_dict = {p_pathway_name: genes_in_pathway}

        q_pathway_name = "SPLAT:::test_pathway:::q"
        if self._q_cache is None:
            pathways_dict[q_pathway_name] = all_genes

        np.random.seed(seed)
        control_pathway_names = []
        for i in range(control_size):
            control_genes = np.random.choice(
                all_genes, len(genes_in_pathway), replace=False
            )
            control_pathway_name = f"SPLAT:::test_pathway:::CONTROL{i}"
            pathways_dict[control_pathway_name] = control_genes
            control_pathway_names.append(control_pathway_name)

        activity_scores_df = self.generate_activity_scores_report(
            pathways_dict=pathways_dict, beta=beta, n_jobs=n_jobs
        ).activity_scores_df()

        p_vector = activity_scores_df[p_pathway_name].to_numpy()
        if self._q_cache is None:
            q_vector = activity_scores_df[q_pathway_name].to_numpy()
            self._q_cache = q_vector
        else:
            q_vector = self._q_cache

        p_control_vectors = activity_scores_df[control_pathway_names].to_numpy().T

        def euclidean_distance(x, y, log=True):
            norm = np.linalg.norm(x - y)
            if log:
                return np.log(norm)
            return norm

        control_distance_distribution = [
            euclidean_distance(p_control, q_vector) for p_control in p_control_vectors
        ]
        observed_distance = euclidean_distance(p_vector, q_vector)
        z = (observed_distance - np.mean(control_distance_distribution)) / np.std(
            control_distance_distribution
        )
        return 2 * min(scipy.stats.norm.cdf(z), 1 - scipy.stats.norm.cdf(z))

    def test_pathway_spatial(
        self,
        pathway: str | None = None,
        genes_in_pathway: list[str] | None = None,
        beta: float = 0.25,
        control_size: int = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Conducts a permutation test to determine if the pathway is
        significantly enriched in the data at certain spatial locations.

        Parameters
        ----------
        pathway : str | None
            Default: None. Name of the pathway to test. If None, genes_in_pathway must
            be provided.

        genes_in_pathway : list[str] | None
            Default: None. List of genes in the pathway to test. If None, pathway must
            be provided. Overrides pathway if not None.

        beta : float
            Default: 0.25. Must be in the interval [0, 1]. Suggested beta < 0.5.

        control_size : int
            Default: 100. Number of control genes to sample for the permutation test.

        seed : int
            Default: 42. Random seed for reproducibility.

        n_jobs : int
            Default: -1. Number of parallel jobs to run. If -1, uses all available CPUs.

        Returns
        -------
        pd.DataFrame
            A DataFrame with n_obs rows and 3 columns: 'x', 'y', and 'p-vals'. The 'x'
            and 'y' columns contain the spatial coordinates of the observations. The
            'p-vals' column contains the p-values of the permutation test for each
            observation.
        """
        if pathway is None and genes_in_pathway is None:
            raise ValueError("Both pathway and genes_in_pathway cannot be None.")

        all_genes = self._expression_df.index.to_list()

        if pathway is not None:
            genes_in_pathway = self._pathways_df[
                self._pathways_df[pathway] == 1
            ].index.to_list()
        p_pathway_name = "SPLAT:::test_pathway:::p"
        pathways_dict = {p_pathway_name: genes_in_pathway}

        np.random.seed(seed)
        control_pathway_names = []
        for i in range(control_size):
            control_genes = np.random.choice(
                all_genes, len(genes_in_pathway), replace=False
            )
            control_pathway_name = f"SPLAT:::test_pathway:::CONTROL{i}"
            pathways_dict[control_pathway_name] = control_genes
            control_pathway_names.append(control_pathway_name)

        activity_scores_df = self.generate_activity_scores_report(
            pathways_dict=pathways_dict, beta=beta, n_jobs=n_jobs
        ).activity_scores_df()

        p_cap = activity_scores_df[p_pathway_name].to_numpy()
        p_matrix = activity_scores_df[control_pathway_names].to_numpy().T
        prob_greater = np.sum(p_matrix > p_cap, axis=0) / len(p_matrix)
        # p_vals = 2 * np.minimum(prob_greater, 1 - prob_greater)
        p_vals = prob_greater
        output = self._locations_df[["x", "y"]].join(
            pd.DataFrame({"p-vals": p_vals}, index=self._locations_df.index)
        )
        return output

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
            Default: None. DataFrame containing spatial coordinates of observations.
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
        self._gene_expression_df and index of self._location_df are equivalent).
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

    def _verify_location_df(self):
        """
        Checks that the format of location_df is reasonable.
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
                    f"{len(common_genes)} genes remain."
                )

        print_removal_info(n_genes_removed_pathway, "expression")
        print_removal_info(n_genes_removed_expression, "pathway")

        print_wrapped(
            f"Identified {len(common_genes)} common genes in the pathway "
            "and expression data."
        )

        common_genes = list(common_genes)
        self._pathways_df = self._pathways_df.loc[common_genes]
        self._expression_df = self._expression_df.loc[common_genes]

    def _force_common_observation(self):
        """
        Finds the common subset of observations. Then, indexes the location and
        expression dataframes to only include the common observations.
        """

        def get_obs_set(df, attr: str) -> set[str]:
            return set(getattr(df, attr))

        obs_location_df = get_obs_set(self._locations_df, "index")
        obs_expression_df = get_obs_set(self._expression_df, "columns")
        common_obs = obs_location_df.intersection(obs_expression_df)

        n_obs_removed_location = len(obs_location_df - common_obs)
        n_obs_removed_expression = len(obs_expression_df - common_obs)

        def print_removal_info(n_removed: int, data_type: str):
            if n_removed > 0:
                print_wrapped(
                    f"Removed {n_removed} observations not found in {data_type} data. "
                    f"{len(common_obs)} observations remain."
                )

        print_removal_info(n_obs_removed_location, "expression")
        print_removal_info(n_obs_removed_expression, "location")

        print_wrapped(
            f"Identified {len(common_obs)} common observations in the location "
            "and expression data."
        )

        common_obs_list: list[str] = list(common_obs)
        self._locations_df = self._locations_df.loc[common_obs_list]
        self._expression_df = self._expression_df[common_obs_list]
