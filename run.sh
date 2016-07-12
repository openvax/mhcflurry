#!/bin/bash
# run jupyter notebook with mhcflurry installed
# navigate to localhost:8888 after running this
# to stop: docker stop mhcflurry
mkdir -p sharedfolder
docker run --name=mhcflurry -it -p 8888:8888 -p 6006:6006 -v sharedfolder:/root/sharedfolder -d hammerlab/mhcflurry:latest bash run_jupyter.sh