Introduction
============

.. currentmodule::gesso

GESSO is a Python package designed for the analysis of spatial transcriptomics
expression data at the gene set/pathway level.
Given a user-provided gene set/pathway, GESSO computes a gene set activity score (GAS)
for each spatial spot in the dataset.


Installation
------------
Install GESSO into a new Python environment.
Your environment should have Python 3.12, 3.13, or 3.14, as well as pip and git installed.

.. code-block:: bash

    git clone https://github.com/YMa-lab/GESSO.git
    cd GESSO
    pip install .
    cd ..


Quick Start
-----------
GESSO processes spatial transcriptomics data along with a user-defined gene set (pathway) and outputs a gene set activity score (GAS) for each spatial spot.

Suppose your spatial transcriptomics dataset contains counts for :math:`G` genes across :math:`N` spots. You’ll need to prepare an :math:`N \times G` expression ``pd.DataFrame`` as well as an :math:`N \times 2` locations ``pd.DataFrame``. The indices of the two DataFrames should match. The locations DataFrame should contain two columns named ``x`` and ``y``.

.. code-block:: python

    import pandas as pd
    from gesso import GESSO

    # load data
    expression_df: pd.DataFrame = ...
    locations_df: pd.DataFrame = ...

    # initialize a GESSO model
    model = GESSO(
        expression_df=expression_df,
        locations_df=locations_df,
        k=20,   # increase k to increase spatial smoothing effect
        normalize_counts_method="normalize-log1p"   # optional, use for raw data
    )

    # compute gene set activity scores
    gas_report = model.compute_gas(
        genesets_dict={
            "example_geneset_1": ["gene1", "gene2", "gene3"],
            "example_geneset_2": ["gene4", "gene5", "gene6"],
        },
        n_jobs=2        # number of parallel jobs
    )
    gas_df = gas_report.gas_df()    # returns N by n_genesets df
    gas_df.to_csv("gas_output.csv")

    # test whether each spot exhibits significantly elevated gene set activity
    htest_report = model.htest_elevated_gas(
        geneset="example_geneset_1",
        genes_in_geneset=["gene1", "gene2", "gene3"],
        n_permutations=200,
        n_jobs=8
    )
    htest_df = htest_report.htest_df()  # returns N by 4 df w/ columns 'x', 'y', 'p', 'gas'
    htest_df.to_csv("htest_output.csv")

Sparse expression data
----------------------

For sparse counts, create a zero-filled pandas sparse DataFrame to avoid a dense
copy of the expression matrix. Normalization preserves sparse storage, and gene
centering is applied implicitly during computation:

.. code-block:: python

    expression_df = pd.DataFrame.sparse.from_spmatrix(
        sparse_counts, index=spot_ids, columns=gene_names
    )

Permutation memory and optional storage
---------------------------------------

Permutation tests process random gene sets in bounded batches and retain only
per-spot exceedance counts by default. ``batch_size`` controls the maximum number
of null score vectors collected at once; the default is twice the worker count.
Changing it does not change the sampled gene sets or the p-value calculation.

To retain null scores, provide a new output directory:

.. code-block:: python

    htest_report = model.htest_elevated_gas(
        genes_in_geneset=["gene1", "gene2", "gene3"],
        n_permutations=500,
        n_jobs=4,
        batch_size=8,
        null_scores_dir="results/pathway_null_scores",
    )
    null_scores = htest_report.null_scores()  # read-only NumPy memory map
    first_permutation = null_scores[0]

``null_scores.npy`` has shape ``(n_permutations, n_spots)`` and float64 values.
The directory also contains ``observed_scores.npy``, ``locations.json`` (spot
labels and coordinates, readable with ``pd.read_json(..., orient="table")``),
``sampled_genes.jsonl`` (one gene list per permutation, numbered from zero), and
``metadata.json`` (parameters, score layout, and completion status).
The score columns follow the stored location order. Existing directories are
refused. Interrupted runs remain marked incomplete and cannot be opened through
``null_scores()``; automatic resumption is not supported.

Saving requires approximately ``8 * n_permutations * n_spots`` bytes for null
scores, plus metadata. Without ``null_scores_dir``, no output files are created.
The test retains the existing strict comparison and p-value definition:
the fraction of null scores greater than the observed score at each spot.

