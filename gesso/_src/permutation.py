"""Bounded-memory permutation reduction and optional on-disk score storage."""

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


class _PermutationAccumulator:
    """Retain one observed vector and one integer count per spot."""

    def __init__(self, observed):
        self.observed = np.asarray(observed, dtype=np.float64)
        self.exceedances = np.zeros(self.observed.shape, dtype=np.int64)
        self.n_completed = 0

    def update(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.observed.shape or not np.isfinite(scores).all():
            raise ValueError(
                "Null scores must be finite and aligned to the observed spots."
            )
        # Preserve the existing strict comparison and uncorrected denominator.
        self.exceedances += scores > self.observed
        self.n_completed += 1

    def p_values(self):
        if self.n_completed == 0:
            raise ValueError("At least one permutation is required.")
        return self.exceedances / self.n_completed


class _NullScoreStore:
    """Append rows to an NPY file without mapping or buffering the entire array.

    A fresh directory is required. Metadata marks partial runs as incomplete;
    the report only exposes the scores after all rows have been written.
    """

    def __init__(self, directory, observed, locations, metadata):
        self.directory = Path(directory).expanduser().resolve()
        self.observed = observed
        self.locations = locations
        self.metadata = {
            **metadata,
            "format_version": 1,
            "complete": False,
            "completed_permutations": 0,
            "score_shape": [metadata["n_permutations"], len(observed)],
            "score_dtype": "<f8",
            "score_axes": ["permutation", "spot"],
            "comparison": "null_score > observed_score",
            "p_value_method": "exceedance_count / n_permutations",
        }
        self._files = ExitStack()
        self.n_written = 0

    def _write_metadata(self):
        temp = self.directory / "metadata.json.tmp"
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)
            handle.write("\n")
        temp.replace(self.directory / "metadata.json")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self._write_metadata()
            with (self.directory / "observed_scores.npy").open("xb") as handle:
                np.save(handle, self.observed, allow_pickle=False)
            self.locations[["x", "y"]].to_json(
                self.directory / "locations.json", orient="table", index=True
            )
            self._scores = self._files.enter_context(
                (self.directory / "null_scores.npy").open("xb")
            )
            np.lib.format.write_array_header_2_0(
                self._scores,
                {
                    "descr": "<f8",
                    "fortran_order": False,
                    "shape": tuple(self.metadata["score_shape"]),
                },
            )
            self._genes = self._files.enter_context(
                (self.directory / "sampled_genes.jsonl").open("x", encoding="utf-8")
            )
        except BaseException:
            self._files.close()
            raise
        return self

    def append(self, scores, genes):
        if self.n_written >= self.metadata["n_permutations"]:
            raise ValueError("More null scores were supplied than requested.")
        record = json.dumps({"permutation": self.n_written, "genes": genes})
        np.asarray(scores, dtype="<f8").tofile(self._scores)
        self._genes.write(record + "\n")
        self.n_written += 1

    def checkpoint(self):
        self._scores.flush()
        self._genes.flush()
        self.metadata["completed_permutations"] = self.n_written
        self._write_metadata()

    def __exit__(self, exc_type, exc_value, traceback):
        self._files.close()
        complete = (
            exc_type is None and self.n_written == self.metadata["n_permutations"]
        )
        self.metadata["completed_permutations"] = self.n_written
        self.metadata["complete"] = complete
        self._write_metadata()
        if exc_type is None and not complete:
            raise RuntimeError("The null-score store is incomplete.")
        return False
