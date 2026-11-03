"""AnnData input, sparse computation, and H5AD export regressions."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gesso import GESSO

ad = pytest.importorskip("anndata")


@pytest.fixture
def adata(expression_df, locations_df):
    result = ad.AnnData(
        X=sparse.csr_matrix(expression_df.to_numpy()),
        obs=pd.DataFrame(index=expression_df.index),
        var=pd.DataFrame(index=expression_df.columns),
    )
    result.obsm["spatial"] = locations_df.to_numpy(copy=True)
    return result


@pytest.mark.parametrize("storage", ["dense", "csr", "csc", "csr_array", "csc_array"])
@pytest.mark.parametrize("normalization", ["none", "normalize-log1p"])
def test_constructor_scores_match_dataframe(
    adata, expression_df, locations_df, genesets_dict, storage, normalization
):
    constructors = {
        "dense": np.array,
        "csr": sparse.csr_matrix,
        "csc": sparse.csc_matrix,
        "csr_array": sparse.csr_array,
        "csc_array": sparse.csc_array,
    }
    counts = expression_df.to_numpy()
    adata.X = constructors[storage](counts)
    original = adata.X.copy()
    expected = GESSO(
        expression_df,
        locations_df,
        k=4,
        normalize_counts_method=normalization,
        verbose=False,
    )
    actual = GESSO.from_anndata(
        adata, k=4, normalize_counts_method=normalization, verbose=False
    )
    # beta=0 isolates adapter equivalence from the legacy directed spatial graph.
    kwargs = dict(genesets_dict=genesets_dict, beta=0, n_jobs=1, verbose=False)
    pd.testing.assert_frame_equal(
        actual.compute_gas(**kwargs).gas_df(),
        expected.compute_gas(**kwargs).gas_df(),
        atol=1e-8,
        rtol=1e-8,
    )
    if storage == "dense":
        np.testing.assert_array_equal(adata.X, original)
    else:
        assert (adata.X != original).nnz == 0
        assert all(
            isinstance(dtype, pd.SparseDtype) for dtype in actual._expression_df.dtypes
        )


def test_sparse_path_never_densifies(adata, genesets_dict, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Sparse expression was densified")

    for cls in (sparse.csr_matrix, sparse.csc_matrix):
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    monkeypatch.setattr(ad.AnnData, "to_df", forbidden)
    model = GESSO.from_anndata(
        adata, k=4, normalize_counts_method="normalize-log1p", verbose=False
    )
    report = model.compute_gas(genesets_dict=genesets_dict, n_jobs=1, verbose=False)
    expression = adata.X
    report.write_to_anndata(adata)
    assert adata.X is expression
    assert np.isfinite(adata.obsm["gesso"].to_numpy()).all()


def test_layer_coordinates_and_view(adata, genesets_df):
    adata.layers["counts"] = adata.X.copy()
    adata.X = sparse.csr_matrix(adata.shape)
    adata.obsm["custom"] = adata.obsm["spatial"] + 10
    view = adata[::-1, :]
    model = GESSO.from_anndata(
        view,
        layer="counts",
        spatial_key="custom",
        genesets_df=genesets_df,
        k=4,
        verbose=False,
    )
    assert view.is_view
    assert adata.X.nnz == 0
    assert model._expression_df.sparse.to_coo().nnz > 0
    np.testing.assert_array_equal(
        model._locations_df.to_numpy(copy=True), view.obsm["custom"]
    )
    assert list(model._locations_df.index) == list(view.obs_names)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing_layer", "Expression layer"),
        ("missing_spatial", "Spatial coordinates"),
        ("duplicate_cells", "obs_names"),
        ("duplicate_genes", "var_names"),
        ("coords_3d", "shape"),
        ("coords_nan", "finite"),
        ("expression_nan", "NaN"),
        ("no_expression", "in-memory"),
    ],
)
def test_invalid_input(adata, problem, match):
    kwargs = {}
    if problem == "missing_layer":
        kwargs["layer"] = "missing"
    if problem == "missing_spatial":
        kwargs["spatial_key"] = "missing"
    if problem == "duplicate_cells":
        adata.obs_names = ["same"] * adata.n_obs
    if problem == "duplicate_genes":
        adata.var_names = ["same"] * adata.n_vars
    if problem == "coords_3d":
        adata.obsm["spatial"] = np.ones((adata.n_obs, 3))
    if problem == "coords_nan":
        adata.obsm["spatial"][0, 0] = np.nan
    if problem == "expression_nan":
        adata.X.data[0] = np.nan
    if problem == "no_expression":
        adata.X = None
    with pytest.raises((ValueError, TypeError), match=match):
        GESSO.from_anndata(adata, **kwargs)


def test_wrong_object():
    with pytest.raises(TypeError, match="AnnData"):
        GESSO.from_anndata(pd.DataFrame())


@pytest.fixture
def report(adata, genesets_dict):
    model = GESSO.from_anndata(adata, k=4, verbose=False)
    return model.compute_gas(genesets_dict=genesets_dict, n_jobs=1, verbose=False)


def test_export_alignment_roundtrip_and_preservation(adata, report, tmp_path):
    target = adata[::-1].copy()
    target.obs["annotation"] = "tissue"
    target.uns["other"] = {"keep": "this"}
    expression = target.X
    coordinates = target.obsm["spatial"].copy()
    assert (
        report.write_to_anndata(target, key="pathways", obs_columns=["geneset_A"])
        is None
    )
    pd.testing.assert_frame_equal(
        target.obsm["pathways"], report.gas_df().reindex(target.obs_names)
    )
    pd.testing.assert_series_equal(
        target.obs["pathways:geneset_A"],
        target.obsm["pathways"]["geneset_A"],
        check_names=False,
    )
    assert target.X is expression
    np.testing.assert_array_equal(target.obsm["spatial"], coordinates)
    assert (target.obs["annotation"] == "tissue").all()
    assert target.uns["other"] == {"keep": "this"}
    assert target.uns["pathways"]["settings"]["expression_source"] == "X"
    target.write_h5ad(tmp_path / "results.h5ad")
    restored = ad.read_h5ad(tmp_path / "results.h5ad")
    pd.testing.assert_frame_equal(restored.obsm["pathways"], target.obsm["pathways"])
    assert sparse.issparse(restored.X)
    assert (restored.X != target.X).nnz == 0
    assert restored.uns["pathways"]["settings"]["k"] == 4
    target.obsm["pathways"].iloc[0, 0] = -999
    assert report.gas_df().iloc[-1, 0] != -999


@pytest.mark.parametrize("destination", ["obsm", "uns", "obs"])
def test_collisions_are_atomic_and_overwrite_is_explicit(adata, report, destination):
    if destination == "obsm":
        adata.obsm["gesso"] = np.zeros((adata.n_obs, 2))
    if destination == "uns":
        adata.uns["gesso"] = {"keep": "me"}
    if destination == "obs":
        adata.obs["gesso:geneset_A"] = -1
    before = adata.copy()
    with pytest.raises(ValueError, match="overwrite"):
        report.write_to_anndata(adata, obs_columns=["geneset_A"])
    assert list(adata.obsm) == list(before.obsm)
    assert list(adata.uns) == list(before.uns)
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    report.write_to_anndata(adata, obs_columns=["geneset_A"], overwrite=True)
    pd.testing.assert_frame_equal(adata.obsm["gesso"], report.gas_df())


def test_export_rejects_mismatched_cells_view_and_invalid_columns(adata, report):
    with pytest.raises(ValueError, match="view"):
        report.write_to_anndata(adata[:])
    target = adata.copy()
    target.obs_names = [f"wrong_{i}" for i in range(target.n_obs)]
    with pytest.raises(ValueError, match="cell IDs"):
        report.write_to_anndata(target)
    with pytest.raises(ValueError, match="cell IDs"):
        report.write_to_anndata(adata[:-1].copy())
    with pytest.raises(ValueError, match="obs_columns"):
        report.write_to_anndata(adata, obs_columns=["missing"])
    assert "gesso" not in adata.obsm
    assert "gesso" not in adata.uns


def test_backed_input_and_export_are_explicitly_unsupported(adata, report, tmp_path):
    path = tmp_path / "backed.h5ad"
    adata.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        with pytest.raises(ValueError, match="Disk-backed"):
            GESSO.from_anndata(backed)
        with pytest.raises(ValueError, match="Disk-backed"):
            report.write_to_anndata(backed)
    finally:
        backed.file.close()


def test_lowres_settings_export(adata, genesets_dict):
    model = GESSO.from_anndata(adata, k=4, verbose=False)
    report = model.compute_gas(
        genesets_dict=genesets_dict,
        compute_method="lowres",
        n_partitions=2,
        partition_method="random",
        partition_seed=7,
        n_jobs=1,
        verbose=False,
    )
    report.write_to_anndata(adata)
    assert adata.uns["gesso"]["settings"]["partition_seed"] == 7
    assert adata.uns["gesso"]["settings"]["n_partitions"] == 2
