# Modeling and interpretation

This note describes the repository implementation. Cite the
[GESSO paper](https://doi.org/10.1038/s41467-026-78316-z) for the method;
check the installed source for numerical details when versions differ.

## Inputs and fitted quantities

The public input has spots in rows and genes in columns. Here,
`G` is the total number of measured genes, `N` is the number of spatial
locations, and gene set `l` contains `G_l` genes. Internally, its expression
matrix `X_l` has **G_l genes × N spots**, after optional count normalization
and per-gene standardization across spots. The rank-one representation
and unit-score constraint are:

```text
X_l ≈ u qᵀ,  ||q||₂ = 1
u: G_l gene contribution weights
q: N spatial activity components
```

In `gLPCA_sparse`, the local variables `X` and `n` correspond to `X_l` and `N`.
The implementation rescales the fitted pair before reporting it: its `u` has
unit norm and its `v` contains the reported GASs, as detailed below. This
changes the scaling convention but preserves the rank-one product.

Each gene set is fitted separately. This is a spatially regularized latent
component, not a supervised predictor, gene-set average, cell-type classifier,
or model of biochemical reaction rates. Gene-set membership supplies the
biological grouping; the data determine weights and the spatial activity pattern.

Normalization and standardization are different operations. The optional
`normalize-log1p` step rescales each spot's total counts to the median total
across spots, then applies log1p. Scoring subsequently centers each gene and
divides by its population standard deviation across the modeled spots. Constant
genes contribute zero standardized variation. For sparse input, centering is
represented implicitly rather than stored as a dense matrix.

## Spatial objective and solver

The graph connects spots to their `k` nearest neighbors in x/y coordinates.
With adjacency `A` and row-degree matrix `D`, the implementation builds
`L = D - A`. Spatial coordinates determine neighbors; no histology image,
cell-type labels, or Squidpy graph is used by this constructor.

Let `λ_n` and `ξ_n` be the expression and graph spectral scales
(called `lmbda` and `xi` in the solver). The implementation applies the following operator to an arbitrary N-vector `z`:

```text
λ_n = largest eigenvalue of X_lᵀX_l
ξ_n = graph spectral scale returned by the largest-eigenvalue routine

M_code z = (1 - β + β/N) z - ((1 - β)/λ_n) X_lᵀ(X_l z) + (β/ξ_n) Lz
```

Here `M_code = M_β + (β/N) I`, with
`M_β = (1 - β)(I - X_lᵀX_l/λ_n) + (β/ξ_n)L`.
The additional identity shift changes eigenvalues but not eigenvectors.
The code calls the Gram operator `G`; this local name does not denote the
total gene count `G`.

It calls the smallest-eigenpair routine on this operator. In the symmetric
graph formulation, this corresponds to minimizing `qᵀM_code q` subject to
`||q||₂ = 1`: retaining expression variation competes with graph roughness.
The identity term is constant under that constraint. The expression Gram
matrix need not be materialized; matrix-vector products provide the same
operator with lower memory use.

Implementation detail: the current k-nearest-neighbor adjacency is directed,
so `L` is not guaranteed symmetric, although the code uses `eigsh`. Do not
describe the implemented graph as automatically symmetrized, or silently change
it while reproducing an analysis. Assess any graph/solver change separately;
the symmetric objective interpretation alone does not establish numerical
guarantees for a nonsymmetric graph.

Given the returned unit vector `q*`, the code computes:

```text
s = ||X_l q*||₂
u_code = X_l q* / s
v_code = s q*
```

Thus `u* = X_l q*` and `q*` give the same rank-one product as
the implementation's `u_code` and `v_code`. The latter are named `u_optimal`
and `v_optimal` in the source, and populate gene contributions and GAS reports.

The gene weights have unit Euclidean norm before any low-resolution averaging.
The common sign of `u` and `v` is ambiguous. The standard scoring path aligns
the score direction with the mean standardized expression across pathway genes.
This is an orientation convention, not an externally calibrated activity scale.

## Choosing parameters

| Parameter | Modeling effect | Practical consequence |
|---|---|---|
| `beta` | Expression/spatial trade-off | `0` gives the leading expression component; larger values favor spatial coherence. Near `1`, expression contributes little and boundaries may disappear. The endpoint can be degenerate. |
| `k` | Number of spatial neighbors | Changes the graph's neighborhood scale. Consider spot density, tissue gaps, and section boundaries. |
| `compute_method` | Full-data or subset-based fitting | Use `cpu` for a full-data reference and `lowres` for larger inputs; assess important patterns when changing methods. |
| `n_partitions` | Number of subsets in `lowres` | More partitions reduce individual problem size but leave fewer spots per fit. |
| `partition_seed` | Subset assignment | Record it for reproducibility and assess sensitivity where conclusions depend on partitioning. |

In `lowres`, stratified k-means first defines spatial strata, then distributes
spots from each stratum across disjoint subsets. Each subset spans the tissue;
the method does not replace spatial clusters with averaged pseudo-spots.
Genes are standardized and graphs are built within each subset. After sign
alignment, scores are placed back at their original spot IDs and gene weights
are averaged across subsets. These scores need not equal a full-data fit, and
the averaged gene weights need not retain unit norm.

## What the outputs support

- Use GAS to describe relative spatial patterns within a fitted gene set.
  Scores are not constrained to `[0, 1]`. A negative value does not, by itself,
  establish biological pathway inhibition.
- Do not interpret score ratios across pathways or independently fitted
  datasets as fold changes. Gene-set size, coverage, preprocessing, and fitted
  component strength affect the scale.
- Large absolute gene weights identify genes strongly involved in the component;
  signed weights indicate direction relative to the score. The report defaults
  to sorting by signed weight, not absolute magnitude. Low weights do not prove
  biological irrelevance, and weights are not differential-expression tests.
- A coherent spatial map can reflect cell composition or technical structure as
  well as biological regulation. Interpret it with the available annotations,
  tissue context, and coverage of the intended gene set.

## Null model and significance

The current permutation test samples same-size random gene sets without
replacement within each draw from the model's retained gene universe. It
recomputes scores using the same expression data, graph, and beta. It does not
shuffle spot coordinates or expression values. Across draws, gene sets may
overlap; sampling is not matched for gene expression level or detection rate.

For spot `i` and `n` control gene sets (`n_permutations` in the API),
the implementation returns:

```text
p_i = count(null_score[r, i] > observed_score[i], r = 1, ..., n) / n
```

This is a one-sided, strict exceedance fraction, without a plus-one correction
or multiple-testing adjustment. A reported zero means no exceedances were
observed in the finite draws; it does not imply zero underlying probability.
Document any downstream multiple-testing procedure separately.

The background universe matters. Supplying a restricted gene membership table
at model construction can remove expression genes before testing; an already
subsetted input also limits the null. Keep the intended background expression
available and supply tested genes explicitly when appropriate. The bundled
334-gene BRCA subset is a workflow example, not a whole-transcriptome background.

## Implementation map

In a GESSO checkout, inspect:

- `gesso/_src/gesso.py`: input alignment, graph construction, scoring modes,
  partition aggregation, and null gene-set sampling.
- `gesso/_src/computation.py`: normalization, implicit standardization, operator,
  eigenvector scaling, and sign alignment.
- `gesso/_src/permutation.py`: strict exceedance counting and optional null storage.
- `docs/gesso_demo_behavior_at_extremes.ipynb`: parameter and coverage examples.
