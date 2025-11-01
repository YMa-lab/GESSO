GESSO :py:mod:`(gesso.GESSO)`
=============================

.. currentmodule:: gesso


`gesso.GESSO` is the main class for the `gesso` Python package.
A GESSO object is initialized with a spatial transcriptomics dataset and 
computes gene set activity scores (GASs) for user-defined gene sets or pathways.

.. autoclass:: gesso.GESSO
    :members:
        __init__,
        compute_gas,
        htest_elevated_gas
