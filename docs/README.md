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
