"""The core import and DataFrame API work without any scverse packages."""

import subprocess
import sys


def test_core_without_optional_dependencies():
    code = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'anndata', 'scanpy', 'squidpy'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockOptional())
import numpy as np
import pandas as pd
from gesso import GESSO
expression = pd.DataFrame(np.arange(80).reshape(10, 8))
locations = pd.DataFrame({'x': np.arange(10), 'y': np.zeros(10)})
model = GESSO(expression, locations, k=2, verbose=False)
report = model.compute_gas(genesets_dict={'example': list(range(8))}, beta=0, n_jobs=1, verbose=False)
assert report.gas_df().shape == (10, 1)
for call in (lambda: GESSO.from_anndata(None), lambda: report.write_to_anndata(None)):
    try:
        call()
    except ImportError as error:
        assert 'gesso-py[anndata]' in str(error)
    else:
        raise AssertionError('Missing dependency was not reported')
assert not {'anndata', 'scanpy', 'squidpy'} & set(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
