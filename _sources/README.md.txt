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
