# MHCflurry 2.3.22

MHCflurry now requires Python 3.11 or later, pandas 3 (`pandas>=3,<4`) and
NumPy 2 (`numpy>=2,<3`), the baseline shared by the openvax libraries. CI
tests Python 3.11 through 3.14. On Python 3.10, pip keeps installing 2.3.21
or earlier.

- The historical data-generation scripts in `downloads-generation/` no longer
  use pandas APIs that pandas 3 removed: `DataFrame.applymap` and
  `fillna(method="ffill")` in `curate_ms_by_pmid.py` (#488), `.ix` in the
  Kim-benchmark curation and `write_validation_data.py`, `Series.iteritems`
  in the presentation training-data builder, and positional `axis` arguments.
  `curate_ms_by_pmid.py` collapses repeated whitespace in HLA strings again:
  pandas 2.0 had made its `str.replace(r"\s+", " ")` a literal replacement.
  A new test runs that script on small in-memory workbooks.
- Python 2 leftovers are gone: `from __future__ import print_function` and
  the undeclared `six` import.
- Ruff targets Python 3.11. The unused `ruff.toml`, which Ruff ignored in
  favor of `.ruff.toml`, is removed (#503).

The prediction models and numerical prediction behavior are unchanged.
