# Changelog

## 0.2.0

Changes since 0.1.0.

### Performance and memory

$N$ is the number of spatial locations, $G$ is the total number of
genes, and gene set $l$ contains $G_l$ genes with expression matrix
$\mathbf{X}_l \in \mathbb{R}^{G_l \times N}$. Here, expression entering the
solver has been standardized per gene. In `gLPCA_sparse`, the matrix and
location count are named `X` and `n`; the public expression DataFrame
has locations in rows.

**Avoid constructing spot-by-spot matrices.** Previously, GESSO explicitly
formed the Gram matrix $\mathbf{X}_l^\top\mathbf{X}_l$ for each gene set.
Although stored in a sparse format, this matrix is generally dense after gene
centering. GESSO also constructed the combined expression-and-spatial objective
matrix.

Now, the eigensolver applies these matrices through their action on an arbitrary
vector $\mathbf{z} \in \mathbb{R}^N$:

$$
(\mathbf{X}_l^\top\mathbf{X}_l)\mathbf{z}
= \mathbf{X}_l^\top(\mathbf{X}_l\mathbf{z}).
$$

This avoids storing the $N \times N$ Gram and objective matrices.
For dense expression, each Gram-vector product takes $O(G_l N)$ work,
instead of multiplying a generally dense Gram matrix in $O(N^2)$;
the $O(G_l N^2)$ Gram construction is also avoided. For scale, one dense
float64 matrix at 100,000 spots would occupy about 80 GB, before intermediates
or parallel workers. This is an algebraic reformulation of the same objective,
not an additional low-resolution approximation. Runtime still depends on
eigensolver convergence. If iteration fails, dense fallback is now limited to
problems with at most 256 spots, avoiding an unexpected large allocation.

**Keep sparse expression sparse.** Previously, scoring converted each gene set
to a dense array before centering and scaling. Now, sparse input remains sparse
during count normalization and log transformation, and centering is applied
implicitly. For this implementation detail, define
$\widetilde{\mathbf{X}}_l$ as expression before gene standardization,
$\boldsymbol{\mu}_l$ as its gene means, and $\mathbf{S}_l$ as the
diagonal inverse-standard-deviation matrix (zero for constant genes). These are
auxiliary preprocessing symbols. With $\mathbf{1}_N$ the
length-$N$ vector of ones:

$$
\mathbf{X}_l\mathbf{z}
= \mathbf{S}_l\left(\widetilde{\mathbf{X}}_l\mathbf{z}
  - \boldsymbol{\mu}_l\,\mathbf{1}_N^\top\mathbf{z}\right).
$$

GESSO therefore stores nonzero expression values and per-gene statistics instead
of materializing all centered values. Expression storage scales with the number
of nonzeros rather than $G_l N$; dense input continues to use dense storage.

**Reuse graph calculations.** Previously, the same spatial graph's spectral
scale was recomputed for every gene set. Low-resolution scoring also rebuilt
each partition's graph for every gene set. Now, graph scales are computed once
and reused; low-resolution partitions, graphs, and scales are cached for the
most recent partition configuration. This removes repeated graph work across
pathways and subsequent calls on the same model.

**Bound permutation memory.** Previously, all $n$ random gene sets were
scored together and their $n \times N$ null scores were retained to compute
p-values. Here $n$ is the number of control gene sets
(`n_permutations` in the API). Now, GESSO
processes at most $b$ permutations per batch (`batch_size`) and
updates a running exceedance count for each spot. Null-score buffering is
$O(bN)$ rather than $O(nN)$, plus $O(N)$ counters; expression
and worker memory are additional. The sampling scheme and p-value calculation
are unchanged:

$$
p_i = \frac{1}{n}\sum_{r=1}^{n}
      \mathbf{1}\{\operatorname{GAS}_i^{(r)}
      > \operatorname{GAS}_i^{\mathrm{observed}}\}.
$$

Here $i$ indexes spatial locations and $r$ indexes control gene
sets; $\operatorname{GAS}$ denotes the reported activity score.

By default, null scores are discarded after updating these counts. Setting
`null_scores_dir` writes them incrementally to disk for later inspection,
using $O(nN)$ disk space without retaining the full array in RAM.

### AnnData integration

* Add `GESSO.from_anndata()` for expression from `adata.X` or a named layer
  and coordinates from `adata.obsm`, preserving sparse input.
* Add `report.write_to_anndata()` to export pathway scores and analysis
  settings, with optional score columns in `adata.obs`.
* Provide optional `anndata` and `tutorial` dependencies. Scanpy and Squidpy
  are used in the tutorial and are not required by the core package.

### Documentation and compatibility

* Add a Scanpy/Squidpy tutorial and a bundled AnnData file using the real
  10x Visium breast cancer dataset.
* Refresh the platform demo outputs and documentation styling.
* Update the citation to the published Nature Communications article and add
  the workflow figure.
* Add a contributing guide and the `gesso_vibe` analysis skill, including
  modeling notes and guidance for AI-assisted analysis.
* Support Python 3.12, 3.13, and 3.14, and replace exact dependency pins with
  minimum supported versions.
* Expand tests for sparse computation, permutation storage, AnnData integration,
  and the tutorial; test both minimum and latest dependencies in CI.
* Distribute GESSO on PyPI as `gesso-py`, retaining the `gesso` Python
  import, with a manual release workflow.

### Acknowledgments

GPT-6-astra, accessed through Codex, assisted with code, tests, and documentation
in the preparation of GESSO v0.2.0.

## 0.1.0

Initial release, including spatial gene set activity scoring, gene contribution
reports, permutation enrichment tests, plotting, and platform tutorials.
