# Contributing to GESSO

Bug reports, documentation improvements, and code contributions are welcome.

## Reporting issues

Open an issue in the repository to report a bug or suggest an improvement. For
bugs, include a small reproducible example, the error message, and your Python
and GESSO versions. Please discuss major changes in an issue before starting work.

## Making changes

1. Fork the repository and create a branch for your changes.
2. Use Python 3.12, 3.13, or 3.14 and install the package from the repository root:

   ```bash
   python -m pip install -e ".[test]"
   ```

3. Keep changes focused and follow the existing code style. Update documentation
   and add or update tests when changing behavior.
4. Run the tests:

   ```bash
   python -m pytest
   ```

5. Open a pull request describing what changed, why, and how you tested it.
   Mention any changes to numerical results or algorithm behavior.

Please keep discussions respectful and constructive.

## Dependencies

Runtime dependencies use tested minimum versions (`>=`) in `pyproject.toml`.
CI tests both the minimum direct dependencies and the newest compatible releases
on all supported Python versions. When changing a minimum, check both environments.
Add an upper bound only for a known incompatibility.

For a reproducible analysis, record the exact versions in your own environment
(for example, with `python -m pip freeze > requirements-analysis.txt`). The
documentation build tools are pinned separately in `docs/requirements.txt`.
