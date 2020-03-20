#!/bin/bash

make doctest
RETVAL=$?
echo doctest returned $RETVAL
cat _build/doctest/output.txt
exit $RETVAL
