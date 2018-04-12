#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
try:
    from setuptools import setup
except:
    from distutils.core import setup

import sys
if sys.version_info < (3, 4):
    raise 'must use Python version 3.4 or higher'

readme = """TBModels is a tool for reading, creating and modifying tight-binding models."""

with open('./tbmodels/_version.py', 'r') as f:
    match_expr = "__version__[^'" + '"]+([' + "'" + r'"])([^\1]+)\1'
    version = re.search(match_expr, f.read()).group(2)

EXTRAS_REQUIRE = {
    'kwant': ['kwant'],
    'dev': [
        'pytest', 'yapf==0.20', 'pythtb', 'pre-commit', 'sphinx',
        'sphinx-rtd-theme==0.2.4'
    ]
}
EXTRAS_REQUIRE['dev'] += EXTRAS_REQUIRE['kwant']

setup(
    name='tbmodels',
    version=version,
    url='http://z2pack.ethz.ch/tbmodels',
    author='Dominik Gresch',
    author_email='greschd@gmx.ch',
    description='Reading, creating and modifying tight-binding models.',
    install_requires=[
        'numpy', 'scipy', 'h5py', 'fsc.export', 'symmetry-representation',
        'click', 'bands-inspect', 'fsc.hdf5-io>=0.2.0'
    ],
    extras_require=EXTRAS_REQUIRE,
    long_description=readme,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English', 'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    entry_points='''
        [console_scripts]
        tbmodels=tbmodels._cli:cli
    ''',
    license='GPL',
    packages=['tbmodels', 'tbmodels._ptools']
)
