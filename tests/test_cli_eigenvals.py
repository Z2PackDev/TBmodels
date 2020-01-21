#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the 'eigenvals' command.
"""

import os
import tempfile

import numpy as np
import bands_inspect as bi
from click.testing import CliRunner

from tbmodels._cli import cli


def test_cli_eigenvals(sample):
    """
    Test the 'eigenvals' command.
    """
    samples_dir = sample('cli_eigenvals')
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli, [
                'eigenvals', '-o', out_file.name, '-k',
                os.path.join(samples_dir, 'kpoints.hdf5'), '-i',
                os.path.join(samples_dir, 'silicon_model.hdf5')
            ],
            catch_exceptions=False
        )
        print(run.output)
        res = bi.io.load(out_file.name)
    reference = bi.io.load(os.path.join(samples_dir, 'silicon_eigenvals.hdf5'))
    np.testing.assert_allclose(bi.compare.difference.calculate(res, reference), 0, atol=1e-10)
