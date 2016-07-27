#!/bin/bash

set -x 
set -e

TAR_PATH=$1
TAR_FILE=$2
JOB=$3

cd $TAR_PATH
#tar xzvf trainer-1.0.tar.gz
tar xzvf $TAR_FILE

cd ./trainer-1.0

python -m trainer.task --job="$JOB"
