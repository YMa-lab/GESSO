"""Run the actual documentation workflow on the bundled real BRCA dataset."""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

ROOT = Path(__file__).resolve().parents[1]


def test_real_brca_anndata_matches_original_data():
    adata = ad.read_h5ad(ROOT / "demo_data" / "10xVisiumBRCA.h5ad")
    counts = pd.read_pickle(ROOT / "demo_data" / "10xVisiumBRCA-count_data.pkl")
    locations = pd.read_pickle(ROOT / "demo_data" / "10xVisiumBRCA-location_data.pkl")
    pathways = pd.read_pickle(ROOT / "demo_data" / "10xVisiumBRCA-pathways_data.pkl")
    assert adata.shape == (2518, 334)
    assert list(adata.obs_names) == list(counts.index)
    assert list(adata.var_names) == list(counts.columns)
    assert sparse.issparse(adata.X)
    assert (adata.X != sparse.csr_matrix(counts.to_numpy())).nnz == 0
    assert (adata.layers["counts"] != adata.X).nnz == 0
    np.testing.assert_array_equal(
        adata.obsm["spatial"],
        locations.loc[counts.index, ["array_col", "array_row"]].to_numpy(),
    )
    pd.testing.assert_frame_equal(
        adata.varm["pathway_membership"],
        pathways.loc[counts.columns],
        check_index_type=False,
        check_column_type=False,
        check_names=False,
    )
    assert set(adata.obs["library_id"]) == {"BRCA"}
    assert list(adata.var["gene_symbols"]) == list(adata.var_names)
    assert set(adata.var["feature_types"]) == {"Gene Expression"}
    spatial = adata.uns["spatial"]["BRCA"]
    assert list(spatial["metadata"]["coordinate_order"]) == ["array_col", "array_row"]
    assert "images" not in spatial and "scalefactors" not in spatial
    for name, filename in adata.uns["dataset"]["source_files"].items():
        assert (
            hashlib.sha256((ROOT / "demo_data" / filename).read_bytes()).hexdigest()
            == adata.uns["dataset"]["source_sha256"][name]
        )


def test_documented_workflow_and_dataframe_equivalence(tmp_path, monkeypatch):
    pytest.importorskip("scanpy")
    pytest.importorskip("squidpy")
    from gesso import GESSO

    monkeypatch.chdir(ROOT / "docs")
    monkeypatch.setenv("GESSO_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    notebook = json.loads((ROOT / "docs" / "anndata_tutorial.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                warnings.simplefilter("error", FutureWarning)
                exec(
                    compile("".join(cell["source"]), "anndata_tutorial.ipynb", "exec"),
                    namespace,
                )
    result = ad.read_h5ad(tmp_path / "10xVisiumBRCA-gesso.h5ad")
    expression = pd.read_pickle(ROOT / "demo_data" / "10xVisiumBRCA-count_data.pkl")
    locations = pd.read_pickle(
        ROOT / "demo_data" / "10xVisiumBRCA-location_data.pkl"
    ).rename(columns={"array_col": "x", "array_row": "y"})
    model = GESSO(
        expression,
        locations,
        k=6,
        normalize_counts_method="normalize-log1p",
        verbose=False,
    )
    reference = model.compute_gas(
        genesets_dict=namespace["pathways"], beta=0.33, n_jobs=1, verbose=False
    ).gas_df()
    pd.testing.assert_frame_equal(
        result.obsm["gesso"],
        reference,
        rtol=1e-6,
        atol=1e-6,
        check_index_type=False,
        check_column_type=False,
    )
    assert sparse.issparse(result.X)
    assert (result.X != sparse.csr_matrix(expression.to_numpy())).nnz == 0
    assert set([namespace["cancer"], namespace["immune"]]).issubset(result.obs.columns)
    import matplotlib.pyplot as plt

    plt.close("all")
