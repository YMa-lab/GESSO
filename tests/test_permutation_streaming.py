"""Streaming equivalence, bounded batching, and optional storage contracts."""

import importlib
import json
import random
from unittest.mock import patch

from joblib import Parallel
import numpy as np
import pandas as pd
import pytest

from gesso import GESSO, PermutationTestReport
from gesso._src.permutation import _PermutationAccumulator

model_module = importlib.import_module("gesso._src.gesso")


def materialized_reference(model, genes, n_permutations=7, seed=17):
    rng = random.Random(seed)
    universe = sorted(model._expression_df.index.to_list())
    samples = [rng.sample(universe, len(genes)) for _ in range(n_permutations)]
    sets = {"observed": genes, **{f"null_{i}": g for i, g in enumerate(samples)}}
    scores = model.compute_gas(
        genesets_dict=sets, n_jobs=1, store_gene_contributions=False
    ).gas_df()
    observed = scores["observed"].to_numpy()
    null = scores[[f"null_{i}" for i in range(n_permutations)]].to_numpy().T
    return observed, null, (null > observed).sum(axis=0) / n_permutations, samples


@pytest.mark.parametrize(
    "batch_size,n_jobs", [(1, 1), (3, 1), (3, 2), (None, 2), (20, 1)]
)
def test_streaming_matches_materialized_reference(
    gesso_model, genesets_dict, batch_size, n_jobs
):
    genes = genesets_dict["geneset_A"]
    observed, _, expected, _ = materialized_reference(gesso_model, genes)
    report = gesso_model.htest_elevated_gas(
        geneset="geneset_A",
        genes_in_geneset=genes,
        n_permutations=7,
        seed=17,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    np.testing.assert_array_equal(report.htest_df()["gas"], observed)
    np.testing.assert_array_equal(report.htest_df()["p"], expected)
    assert list(report.htest_df()) == ["x", "y", "gas", "p"]


def test_accumulator_uses_strict_comparison_and_empirical_denominator():
    accumulator = _PermutationAccumulator(np.array([1.0, 2.0, 3.0]))
    accumulator.update(np.array([1.0, 3.0, 4.0]))
    accumulator.update(np.array([0.0, 2.0, 5.0]))
    np.testing.assert_array_equal(accumulator.p_values(), [0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="finite and aligned"):
        accumulator.update(np.array([np.nan, 0, 0]))


def test_default_writes_no_files_and_preserves_global_rng(
    gesso_model, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    state = random.getstate()
    report = gesso_model.htest_elevated_gas(
        geneset="geneset_A", n_permutations=3, n_jobs=1
    )
    assert random.getstate() == state
    assert list(tmp_path.iterdir()) == []
    assert report.null_scores_dir is None
    with pytest.raises(ValueError, match="not saved"):
        report.null_scores()


def test_parallel_submissions_are_bounded(gesso_model, monkeypatch):
    sizes = []

    class TrackingParallel(Parallel):
        def __call__(self, tasks):
            tasks = list(tasks)
            sizes.append(len(tasks))
            return super().__call__(tasks)

    monkeypatch.setattr(model_module, "Parallel", TrackingParallel)
    with patch.object(
        gesso_model, "compute_gas", wraps=gesso_model.compute_gas
    ) as compute:
        gesso_model.htest_elevated_gas(
            geneset="geneset_A",
            n_permutations=11,
            batch_size=3,
            n_jobs=2,
        )
    # One observed score, followed by bounded null batches; no all-null report.
    assert sizes == [1, 3, 3, 3, 2]
    assert compute.call_count == 1
    assert compute.call_args.kwargs["store_gene_contributions"] is False


@pytest.mark.parametrize("sparse_input", [False, True])
def test_disk_scores_reproduce_results_and_preserve_spot_order(
    expression_df, locations_df, genesets_df, genesets_dict, tmp_path, sparse_input
):
    if sparse_input:
        expression_df = expression_df.astype(pd.SparseDtype("float64", 0))
    model = GESSO(
        expression_df, locations_df.iloc[::-1], genesets_df, k=4, verbose=False
    )
    observed, null, p_values, samples = materialized_reference(
        model, genesets_dict["geneset_A"]
    )
    output = tmp_path / "null_scores"
    report = model.htest_elevated_gas(
        geneset="geneset_A",
        n_permutations=7,
        seed=17,
        batch_size=3,
        n_jobs=2,
        null_scores_dir=output,
    )
    saved = report.null_scores()
    assert isinstance(saved, np.memmap)
    assert not saved.flags.writeable
    assert saved.shape == (7, len(locations_df))
    np.testing.assert_allclose(saved, null, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(report.htest_df()["p"], p_values)
    np.testing.assert_array_equal(np.load(output / "observed_scores.npy"), observed)
    np.testing.assert_array_equal((saved > observed).sum(axis=0) / 7, p_values)
    locations = pd.read_json(output / "locations.json", orient="table")
    pd.testing.assert_frame_equal(locations, model._locations_df[["x", "y"]])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["complete"] is True
    assert metadata["completed_permutations"] == 7
    assert metadata["score_axes"] == ["permutation", "spot"]
    assert metadata["seed"] == 17
    records = [
        json.loads(line)
        for line in (output / "sampled_genes.jsonl").read_text().splitlines()
    ]
    assert [r["genes"] for r in records] == samples
    assert [r["permutation"] for r in records] == list(range(7))


def test_existing_directory_is_not_overwritten(gesso_model, tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep me")
    with patch.object(gesso_model, "compute_gas") as compute:
        with pytest.raises(FileExistsError):
            gesso_model.htest_elevated_gas(
                geneset="geneset_A", null_scores_dir=tmp_path
            )
    compute.assert_not_called()
    assert marker.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [marker]


def test_failed_worker_leaves_incomplete_store(gesso_model, tmp_path, monkeypatch):
    original = model_module.gLPCA_sparse

    def fail_third_null(*args, **kwargs):
        if kwargs["geneset_name"] == "random_geneset_3":
            raise RuntimeError("deliberate worker failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "gLPCA_sparse", fail_third_null)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="deliberate"):
        gesso_model.htest_elevated_gas(
            geneset="geneset_A",
            n_permutations=5,
            batch_size=2,
            n_jobs=1,
            null_scores_dir=output,
        )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["complete"] is False
    assert metadata["completed_permutations"] == 2
    report = PermutationTestReport("geneset_A", pd.DataFrame(), null_scores_dir=output)
    with pytest.raises(ValueError, match="incomplete"):
        report.null_scores()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("n_permutations", 0),
        ("n_permutations", -1),
        ("n_permutations", 2.5),
        ("n_permutations", True),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 2.5),
        ("batch_size", True),
    ],
)
def test_invalid_sizes_fail_before_writing(gesso_model, tmp_path, parameter, value):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=parameter):
        gesso_model.htest_elevated_gas(
            geneset="geneset_A", null_scores_dir=output, **{parameter: value}
        )
    assert not output.exists()


def test_target_name_cannot_collide_with_null_names(gesso_model, genesets_dict):
    genes = genesets_dict["geneset_A"]
    expected = gesso_model.htest_elevated_gas(
        genes_in_geneset=genes,
        n_permutations=5,
        n_jobs=1,
    ).htest_df()
    actual = gesso_model.htest_elevated_gas(
        geneset="random_geneset_1",
        genes_in_geneset=genes,
        n_permutations=5,
        n_jobs=1,
    ).htest_df()
    pd.testing.assert_frame_equal(actual, expected)
