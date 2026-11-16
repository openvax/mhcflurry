# MHCflurry 2.3.29

Remove the unused pylint configuration and the second, Conda-based Python
provisioning step from CI. The workflow uses the interpreter selected by
actions/setup-python and installs the same pip dependencies. Model downloads,
tests and predictions are unchanged.

This finishes the remaining configuration cleanup in #418.
