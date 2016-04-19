#!/usr/bin/env bash
DATA_DIR=`python -c "import mhcflurry; print(mhcflurry.paths.CLASS1_DATA_DIRECTORY)"`
echo "$DATA_DIR"