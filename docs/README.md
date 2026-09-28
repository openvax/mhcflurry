---
orphan: true
---

# MHCflurry documentation

To generate Sphinx documentation, from this directory run:

```shell
pip install -r requirements.txt  # first build only
make generate html
```

Documentation is written to `_build/html`. These files should not be checked
into the source branch. Pull requests build and test the docs; merges to
`master` publish them to the `gh-pages` branch.

To test example code:

```shell
make doctest
```

See `_build/doctest` for detailed output.

For a strict build, use `make generate html doctest SPHINXOPTS="-W --keep-going"`.
The API page includes generated reStructuredText inside an `eval-rst` block;
the three subpackage wrapper pages do the same using native RST includes.
Do not change the API include to a plain Markdown include: the build can then
silently omit the actual class and method documentation. The `api_coverage`
extension fails builds that are missing representative public API entries.
The legacy `local_parallelism` re-export module is excluded from generation;
its canonical API is documented under `mhcflurry.parallelism`.

The shared percentile guide also has executable Python examples and CLI
templates checked by `test/test_percentile_documentation.py`. Keep its
equations, defaults, and compatibility guidance aligned with the predictor
docstrings and command help when changing calibration behavior.
