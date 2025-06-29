SPLAT :py:mod:`(splat.SPLAT)`
=============================

.. currentmodule:: splat


`splat.SPLAT` is the main class for the `splat` Python package.
A SPLAT object is initialized with a spatial transcriptomics dataset and 
computes pathway activity scores (PAS) for user-defined gene sets or pathways.

.. autoclass:: splat.SPLAT
    :members:
        __init__,
        compute_pas,
        htest_elevated_pas
