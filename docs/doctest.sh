#!/bin/bash

sphinx-build -b doctest -d _build/doctrees . _build/doctest
RETVAL=$?
echo doctest returned $RETVAL
cat _build/doctest/output.txt
exit $RETVAL
