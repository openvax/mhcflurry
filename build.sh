#!/bin/bash
docker build -t hammerlab/mhcflurry . ;
docker tag -f hammerlab/mhcflurry hammerlab/mhcflurry:latest ;