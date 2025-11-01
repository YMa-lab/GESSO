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
Your environment should have Python 3.10 or later, as well as pip and git installed.

.. code-block:: bash

    git clone https://github.com/YMa-lab/GESSO.git
    cd gesso
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
        control_size=200,
        n_jobs=8
    )
    htest_df = htest_report.htest_df()  # returns N by 4 df w/ columns 'x', 'y', 'p', 'gas'
    htest_df.to_csv("htest_output.csv")


