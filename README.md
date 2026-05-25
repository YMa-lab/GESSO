# GESSO: Spatial Gene Set Activity Analysis

GESSO (Gene sEt activity Score analysis with Spatial lOcation) is a computational method for 
quantifying the expression of gene sets for spatial transcriptomics data.

This repository is the official Python implementation of GESSO. Please visit the [GESSO tutorial](https://gesso-malab.readthedocs.io/) for more information.

### Installation
We recommend installing GESSO in a new Python environment. You can create a new Python environment through [conda](https://www.anaconda.com/docs/getting-started/miniconda/main):
```bash
conda create -n gesso python=3.12 -y
conda activate gesso
```

GESSO requires Python version 3.12. The following script installs GESSO into your Python environment:
```bash
git clone https://github.com/YMa-Lab/GESSO.git
cd gesso
pip install .
cd ..
```

### Quick start
GESSO processes spatial transcriptomics data along with a user-defined gene set/pathway and outputs a gene set activity score (GAS) for each spatial location.

Let's say your spatial transcriptomics dataset contains counts for $G$ genes across $N$ spots. You'll need to prepare an $N \times G$ expression `pd.DataFrame` as well as an $N \times 2$ locations `pd.DataFrame`. The indices of the two DataFrames must match. The locations DataFrame must contain two columns named `x` and `y`.

```python
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
```
### Logging
GESSO uses Python's standard `logging` module under the `gesso.*` hierarchy. By
default the package is silent (a `NullHandler` is installed on the root
`gesso` logger). Configure output via the `gesso.logging` submodule:

```python
from gesso import logging as glog

glog.enable()                          # print INFO messages to stderr
glog.set_level("DEBUG")                # show debug-level messages as well
glog.silence_per_geneset()             # mute per-geneset progress, keep summaries
glog.unsilence_per_geneset()           # re-enable per-geneset progress
handler = glog.add_file_handler("gesso.log", level="DEBUG")  # also log to file
glog.remove_handler(handler)           # detach when done
glog.disable()                         # back to silent
```

Logger hierarchy:

| Logger                      | What it emits                                       |
| --------------------------- | --------------------------------------------------- |
| `gesso`                     | Root; configure here to affect everything           |
| `gesso.init`                | Preprocessing / model initialization messages       |
| `gesso.compute`             | High-level activity-score computation summaries     |
| `gesso.compute.geneset`     | Per-geneset progress and timing (chatty in workers) |

Each public method that emits log messages also accepts `verbose=True/False` as
a per-call override. Passing `verbose=False` suppresses **all** messages for
that call (including worker output), independent of logger configuration. The
`verbose` argument on `GESSO(...)` sets the default for all calls on that
model instance.

If you have your own application logging configured, you can also attach
handlers directly via stdlib: `logging.getLogger("gesso").addHandler(...)`.

### Tutorial
For the step-by-step tutorial, please refer to: [GESSO tutorial](https://gesso-malab.readthedocs.io/)
