"""Optional AnnData adapters; importing GESSO does not import anndata."""

import numpy as np
import pandas as pd
from scipy import sparse


def validate_anndata(adata, *, writing=False):
    try:
        from anndata import AnnData
    except ImportError as exc:
        raise ImportError(
            "AnnData support requires the optional dependency: "
            "install 'gesso-py[anndata]' (or pip install '.[anndata]' from source)."
        ) from exc
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an anndata.AnnData object.")
    if adata.isbacked:
        raise ValueError(
            "Disk-backed AnnData is not supported. Use adata.to_memory() only "
            "if the expression data fits in memory."
        )
    if writing and adata.is_view:
        raise ValueError("Cannot export to an AnnData view; use adata.copy() first.")
    for name in ("obs_names",) if writing else ("obs_names", "var_names"):
        index = getattr(adata, name)
        if not index.is_unique or index.hasnans or any(not str(x) for x in index):
            raise ValueError(f"adata.{name} must contain unique, nonempty identifiers.")


def expression_and_locations(adata, *, layer, spatial_key):
    validate_anndata(adata)
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Expression layer {layer!r} is not present in adata.layers.")
    matrix = adata.X if layer is None else adata.layers[layer]
    if not isinstance(matrix, np.ndarray) and not sparse.issparse(matrix):
        raise TypeError(
            "Expression must be an in-memory NumPy array or SciPy sparse matrix."
        )
    if matrix.dtype.kind not in "biuf":
        raise ValueError("Expression must contain real numeric values.")
    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN or infinite values.")
    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates {spatial_key!r} are not present in adata.obsm."
        )
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape != (adata.n_obs, 2):
        raise ValueError(
            "Spatial coordinates must have shape (n_obs, 2), with x and y columns."
        )
    if coords.dtype.kind not in "iuf" or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must contain finite real numeric values.")
    if sparse.issparse(matrix):
        expression = pd.DataFrame.sparse.from_spmatrix(
            sparse.csr_matrix(matrix), index=adata.obs_names, columns=adata.var_names
        )
    else:
        expression = pd.DataFrame(
            matrix, index=adata.obs_names, columns=adata.var_names
        )
    locations = pd.DataFrame(coords, index=adata.obs_names, columns=["x", "y"])
    return expression, locations


def write_scores(adata, scores, metadata, *, key, obs_columns, overwrite):
    """Validate every destination before updating the supplied AnnData."""
    from copy import deepcopy
    from gesso import __version__

    validate_anndata(adata, writing=True)
    if not isinstance(key, str) or not key or "/" in key:
        raise ValueError("key must be a nonempty string without '/'.")
    if (
        not scores.index.is_unique
        or len(scores) != adata.n_obs
        or set(scores.index) != set(adata.obs_names)
    ):
        raise ValueError(
            "Report and AnnData must contain exactly the same unique cell IDs."
        )
    if not scores.columns.is_unique or any(
        not isinstance(name, str) or not name or "/" in name for name in scores.columns
    ):
        raise ValueError(
            "Pathway names must be unique nonempty strings without '/' for H5AD export."
        )
    if isinstance(obs_columns, str):
        raise TypeError(
            "obs_columns must be a sequence of pathway names, not a string."
        )
    selected = [] if obs_columns is None else list(obs_columns)
    if len(set(selected)) != len(selected) or any(
        name not in scores for name in selected
    ):
        raise ValueError(
            "obs_columns must contain unique pathway names present in the report."
        )
    columns = [f"{key}:{name}" for name in selected]
    collisions = [name for name in columns if name in adata.obs]
    if not overwrite and (key in adata.obsm or key in adata.uns or collisions):
        raise ValueError(
            "Export would overwrite existing results. Choose another key or set overwrite=True."
        )
    aligned = scores.reindex(adata.obs_names).copy()
    exported_metadata = {
        "gesso_version": __version__,
        "pathways": list(aligned.columns),
        "settings": deepcopy(metadata),
        "obs_columns": columns,
    }
    adata.obsm[key] = aligned
    adata.uns[key] = exported_metadata
    for name, column in zip(selected, columns):
        adata.obs[column] = aligned[name]
