---
name: gesso-analysis
description: Run and interpret GESSO spatial gene set activity analysis on AnnData or expression and coordinate tables. Use for GESSO modeling questions, parameter selection, pathway scoring, spatial plots, permutation enrichment tests, and result export.
---

# GESSO analysis

Use GESSO's Python API for the requested analysis. Locate the user's data and
GESSO installation; the skill directory is not necessarily the repository root.
Install `.[anndata]` from a GESSO checkout for AnnData input, or `.[tutorial]`
when Scanpy/Squidpy plots are needed. Python 3.12–3.14 is supported.

## What GESSO models

GESSO (Gene sEt activity Score analysis with Spatial lOcation) estimates the
spatial activity of predefined gene sets from expression and tissue coordinates.
It fits each gene set separately using a graph Laplacian-regularized rank-one
decomposition: one latent activity pattern across spots and one vector of gene
weights describing that pattern. Spatial regularization encourages neighboring
spots to have similar activity while retaining expression-driven variation.

The modeling flow is:

```text
Expression + gene sets → select pathway genes → standardize each gene
Spatial coordinates   → k-nearest-neighbor graph → graph Laplacian
                     ↓ joint expression/spatial objective
             Spot activity scores + gene contribution weights
                     ↓ optional random-gene-set testing
                     Per-spot enrichment p-values
```

Keep the three outputs distinct:

- **Activity scores (GAS):** a learned, signed spatial pattern for a gene set.
  They are not mean expression, probabilities, or p-values. Interpret high and
  low scores within the fitted analysis; negative values are valid.
- **Gene contributions:** signed metagene weights showing how genes participate
  in the fitted pattern, not causal effects or gene-level significance tests.
- **Enrichment p-values:** evidence that a spot's observed score exceeds scores
  from random gene sets under the implemented null, not proof of a mechanism.

`beta` balances expression and spatial structure; `k` defines the neighborhood.
At `beta=0`, the objective is expression-only; increasing beta emphasizes spatial
coherence and can blur boundaries. The `lowres` mode approximates scoring using
spatially distributed subsets; it is not simply a faster identical solver.

Read [Modeling and interpretation](references/modeling.md) when explaining the
objective, choosing parameters, comparing scores, or interpreting significance.
It includes the implemented equations, scaling/sign conventions, and limits.

## Prepare the input

- For AnnData, use `GESSO.from_anndata(adata, layer=..., spatial_key="spatial")`.
  `layer=None` selects `X`; there is no automatic fallback to `raw` or counts.
  Keep SciPy sparse expression sparse; avoid `.to_df()`, `.toarray()`, and
  `.todense()`. Backed/lazy expression is not supported by this adapter.
- Use unique spot IDs and gene IDs matching the pathway definitions. Check gene
  overlap before scoring; do not silently rename or discard duplicate genes.
  Accept pathways as a dictionary of names to gene lists, or a binary
  gene-by-pathway DataFrame. Avoid restricting expression to highly variable
  genes when that removes pathway members.
- Coordinates must contain finite x/y pairs. Analyze independent tissue sections
  separately so their spatial graphs do not connect unrelated spots.
- Select `normalize_counts_method="normalize-log1p"` for raw counts and `"none"`
  for already normalized, log-transformed expression. Establish the input's
  preprocessing when unclear; do not infer it solely from a layer name.
- For DataFrame input, rows are spots and columns are genes. The coordinate
  table has matching spot IDs and columns `x` and `y`. Sparse pandas input
  must have zero fill values.

## Score and export

Use `compute_gas(genesets_dict=pathways, ...)`. The `cpu` method suits smaller
datasets; `lowres` partitions larger datasets and aggregates scores. Set an
appropriate `n_jobs` for the available resources. Preserve user-selected
parameters; otherwise start with `beta=0.33` and record `k`, normalization,
method, and any partition settings. Low-resolution partitioning uses
`n_partitions`, `partition_method`, and `partition_seed`.

Read scores with `report.gas_df()` and plot with
`report.plot_gas_spatial_map(geneset=..., size=20, figsize=(5, 5))`.
Gene contributions are available through `report.gene_contributions_df(...)`
when retained during computation.

For AnnData, `report.write_to_anndata(adata, key="gesso", obs_columns=[...])`
stores scores in `obsm`, settings in `uns`, and selected pathways in `obs` as
`gesso:pathway`. Export aligns by spot ID and requires exactly the scored cells.
Views need an explicit copy before export; existing result keys require an
explicit overwrite choice. Save to a new output file, preserving the input.

## Run the bundled example

Run this from the GESSO repository root. The real BRCA dataset includes its
pathway table; user datasets may instead supply pathways separately.

```python
from pathlib import Path
import anndata as ad
from gesso import GESSO

adata = ad.read_h5ad("demo_data/10xVisiumBRCA.h5ad")
membership = adata.varm["pathway_membership"]
pathways = {
    name: membership.index[membership[name] == 1].tolist()
    for name in membership.columns
}
model = GESSO.from_anndata(
    adata, layer="counts", k=6,
    normalize_counts_method="normalize-log1p",
)
report = model.compute_gas(
    genesets_dict=pathways, beta=0.33, compute_method="cpu", n_jobs=1,
)
report.write_to_anndata(adata)
output = Path("gesso-results")
output.mkdir()
adata.write_h5ad(output / "brca-gesso.h5ad", compression="gzip")
figure = report.plot_gas_spatial_map(
    geneset="KEGG_PATHWAYS_IN_CANCER", size=20, figsize=(5, 5),
)
figure.savefig(output / "cancer-pathway.png", dpi=150)
```

## Permutation tests

When requested, use `model.htest_elevated_gas(geneset=..., genes_in_geneset=...,
n_permutations=..., seed=..., n_jobs=...)`. This has its own CPU scoring path;
it does not inherit a previous `lowres` configuration. The sampling universe
is the expression genes retained in the model. Avoid narrowing that universe
to pathway genes unintentionally; the bundled BRCA file contains only 334 genes.

`batch_size` bounds permutation batches. `null_scores_dir` optionally writes
null scores to a new directory; this does not make input expression disk-backed.
Retrieve results with `htest_df()` and saved null scores with `null_scores()`.
The `p` column contains unadjusted permutation p-values, not FDR values.

## Check the result

Check spot alignment, output dimensions, finite scores, and reported gene
coverage. Retain warnings relevant to interpretation. Report the actual settings,
package versions, and saved files; distinguish activity scores from statistical
enrichment evidence.

For exact signatures and examples, consult the installed API or the
[API reference](https://gesso-malab.readthedocs.io/en/latest/api_reference.html),
[AnnData tutorial](https://gesso-malab.readthedocs.io/en/latest/anndata_tutorial.html),
and [parameter tuning tutorial](https://gesso-malab.readthedocs.io/en/latest/gesso_demo_behavior_at_extremes.html).
