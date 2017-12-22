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

We have experimented with using the documentation system to generate the mhcflurry
package level README, but this is not currently in use. To build the readme, run:

```
$ make readme
```

This will write `docs/package_readme/readme.generated.txt`. The main
[README.rst](../README.rst) could be symlinked to this file at a later point.

