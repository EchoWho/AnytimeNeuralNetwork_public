#!/bin/bash -e
# File: run-tests.sh
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

DIR=$(dirname $0)
cd $DIR

export TF_CPP_MIN_LOG_LEVEL=2
python -m unittest discover -v
cd ..
# python -m tensorpack.models._test
# segfault for no reason (https://travis-ci.org/ppwwyyxx/tensorpack/jobs/217702985)
