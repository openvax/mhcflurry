#!/usr/bin/env bash

# Download dataset from Kim/Peters 2013 "Dataset size and composition" paper
rm -f bdata.20130222.mhci.public*
wget https://dl.dropboxusercontent.com/u/3967524/bdata.20130222.mhci.public.1.txt
DATA_DIR=`python -c "import mhcflurry; print(mhcflurry.paths.CLASS1_DATA_DIRECTORY)"`
mkdir -p -- "$DATA_DIR"
mv bdata.20130222.mhci.public.1.txt "$DATA_DIR"
