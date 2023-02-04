#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>

# Be verbose, and stop with error as soon there's one
set -ev

pip install -U pip
pip install codecov
pip install -U setuptools wheel poetry

case "$INSTALL_TYPE" in
    dev)
        poetry install
        ;;
    dev_sdist)
        poetry build
        poetry export --only=dev > requirements_test.txt
        pip install dist/*.tar.gz
        pip install -r requirements_test.txt
        ;;
    dev_bdist_wheel)
        poetry build
        poetry export --only=dev > requirements_test.txt
        pip install dist/*.whl
        pip install -r requirements_test.txt
        ;;
esac
