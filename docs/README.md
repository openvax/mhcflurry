# MHCflurry documentation

Due to our use of `sphinxcontrib-autorun2` we unfortunately require Python 2.7
to build to the docs. Python 3 is not supported.

To generate Sphinx documentation, from this directory run:

```
$ pip install -r requirements.txt  # for the first time you generate docs
$ make generate html
```

Documentation is written to the _build/ directory. These files should not be
checked into the repo.

We use the documentation system to generate the mhcflurry package level README.
To build this file, run:

```
$ make readme
```

This will write `docs/package_readme/readme.generated.txt`. The main
[README.rst](../README.rst) is symlinked to this file.

