#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>

# Be verbose, and stop with error as soon there's one
set -ev

cd ${TRAVIS_BUILD_DIR}

case "$INSTALL_TYPE" in
    test)
        pip install .[test]
        ;;
    test_sdist)
        python setup.py sdist
        ls -1 dist/ | xargs -I % pip install dist/%[test]
        ;;
    precommit)
        pip install .[precommit]
        ;;
esac
