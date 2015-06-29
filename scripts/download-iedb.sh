#!/usr/bin/env bash
rm -f mhc_ligand_full*
wget http://www.iedb.org/doc/mhc_ligand_full.zip
unzip mhc_ligand_full.zip
DATA_DIR=`python -c "import mhcflurry; print(mhcflurry.paths.CLASS1_DATA_DIRECTORY)"`
mkdir -p -- "$DATA_DIR"
mv mhc_ligand_full.csv "$DATA_DIR"