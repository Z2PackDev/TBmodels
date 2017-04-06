#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os

import pytest
import tempfile
import numpy as np
import bandstructure_utils as bs
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli
from parameters import SAMPLES_DIR

def test_cli_eigenvals():
    samples_dir = os.path.join(SAMPLES_DIR, 'cli_eigenvals')
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli,
            [
                'eigenvals',
                '-o', out_file.name,
                '-k', os.path.join(samples_dir, 'kpoints.hdf5'),
                '-i', os.path.join(samples_dir, 'silicon_model.hdf5')
            ],
            catch_exceptions=False
        )
        print(run.output)
        res = bs.io.load(out_file.name)
    reference = bs.io.load(os.path.join(samples_dir, 'silicon_eigenvals.hdf5'))
    np.testing.assert_allclose(bs.compare.difference(res, reference), 0, atol=1e-10)
