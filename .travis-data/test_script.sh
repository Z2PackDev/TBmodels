#!/bin/bash

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

# Be verbose, and stop with error as soon there's one
set -ev

case "$TEST_TYPE" in
    tests)
        pytest --cov=tbmodels --cov-config=.coveragerc
        ;;
    precommit)
        pre-commit run --all-files
        ;;
esac
