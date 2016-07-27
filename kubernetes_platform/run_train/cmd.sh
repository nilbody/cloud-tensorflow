#!/bin/bash

set -x
set -e

#TAR_PATH=$1
#TAR_FILE=$2
#JOB=$3

apt-get update -y
apt-get install -y wget

wget https://github.com/tobegit3hub/tensorflow_examples/blob/master/trainer-1.0.tar.gz?raw=true -O trainer-1.0.tar.gz

tar xzvf trainer-1.0.tar.gz

cd ./trainer-1.0

python -m trainer.task
