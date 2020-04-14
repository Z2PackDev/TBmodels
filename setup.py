#!/usr/bin/env python

import warnings

import setuptools

try:
    import fastentrypoints
except ImportError:
    warnings.warn(
        "The 'fastentrypoints' module could not be loaded. "
        "Installed console script will be slower."
    )

if __name__ == "__main__":
    setuptools.setup()
