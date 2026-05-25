from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gesso import logging as glog

glog.disable()


N_ROWS = 6
N_COLS = 10
N_SPOTS = N_ROWS * N_COLS
N_GENES = 80


def _grid_locations(n_rows: int, n_cols: int) -> pd.DataFrame:
    xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    df = pd.DataFrame(
        {"x": xs.ravel().astype(float), "y": ys.ravel().astype(float)},
        index=[f"spot_{i}" for i in range(n_rows * n_cols)],
    )
    return df


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def locations_df() -> pd.DataFrame:
    return _grid_locations(N_ROWS, N_COLS)


@pytest.fixture(scope="session")
def gene_names() -> list[str]:
    return [f"gene_{i}" for i in range(N_GENES)]


@pytest.fixture(scope="session")
def spot_names() -> list[str]:
    return [f"spot_{i}" for i in range(N_SPOTS)]


@pytest.fixture(scope="session")
def genesets_dict(gene_names) -> dict[str, list[str]]:

    return {
        "geneset_A": gene_names[:12],
        "geneset_B": gene_names[12:22],
    }


@pytest.fixture(scope="session")
def genesets_df(gene_names, genesets_dict) -> pd.DataFrame:
    df = pd.DataFrame(0, index=gene_names, columns=list(genesets_dict.keys()))
    for gs, members in genesets_dict.items():
        df.loc[members, gs] = 1
    return df


@pytest.fixture(scope="session")
def expression_df(spot_names, gene_names, locations_df, genesets_dict) -> pd.DataFrame:

    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=0.5, size=(N_SPOTS, N_GENES)).astype(float)

    x = locations_df["x"].to_numpy()
    mid = x.mean()
    left_mask = x < mid
    right_mask = ~left_mask

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    for g in genesets_dict["geneset_A"]:
        counts[left_mask, gene_idx[g]] += rng.poisson(lam=8, size=left_mask.sum())
    for g in genesets_dict["geneset_B"]:
        counts[right_mask, gene_idx[g]] += rng.poisson(lam=8, size=right_mask.sum())

    return pd.DataFrame(counts, index=spot_names, columns=gene_names)


@pytest.fixture()
def gesso_model(expression_df, locations_df, genesets_df):
    from gesso import GESSO

    return GESSO(
        expression_df=expression_df,
        locations_df=locations_df,
        genesets_df=genesets_df,
        k=4,
        verbose=False,
    )


@pytest.fixture()
def gesso_model_no_genesets(expression_df, locations_df):
    from gesso import GESSO

    return GESSO(
        expression_df=expression_df,
        locations_df=locations_df,
        k=4,
        verbose=False,
    )
