# GESSO with AI agents

The [GESSO analysis skill](gesso-analysis/SKILL.md) helps an agent load spatial
data, compute pathway activity, and export results using GESSO's Python API.
It includes a model overview and [modeling notes](gesso-analysis/references/modeling.md)
covering the objective, parameter choices, and interpretation of scores and tests.

## Setup

Install GESSO with AnnData support from PyPI:

```bash
pip install 'gesso-py[anndata]'
```

Install the `gesso-analysis` folder through your agent's skill installer, or point
the agent directly to `gesso_vibe/gesso-analysis/SKILL.md`.
The skill files and bundled demo data are available in the
[repository](https://github.com/YMa-lab/GESSO).

Example request:

> Read gesso_vibe/gesso-analysis/SKILL.md and run the included Visium BRCA
> dataset. Save pathway scores and spatial plots in a new output directory.

For your own data, provide the expression file, gene sets, expression layer,
and output location. The skill uses the existing Python API; no MCP server is
required.
