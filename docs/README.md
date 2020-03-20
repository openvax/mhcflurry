# MHCflurry documentation

To generate Sphinx documentation, from this directory run:

```
$ pip install -r requirements.txt  # for the first time you generate docs
$ make generate html
```

Documentation is written to the _build/ directory. These files should not be
checked into the repo.

To test example code:
```
$ make doctest 
```

Then take a look at _build/doctest for detailed output.

