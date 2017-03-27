#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os

import pytest
import tempfile
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli
from parameters import SAMPLES_DIR

def test_cli_parse(models_close):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        runner.invoke(cli, [
            'symmetrize',
            '-o', out_file.name,
            '-s', os.path.join(SAMPLES_DIR, 'InAs_symmetries.hdf5'),
            '-i', os.path.join(SAMPLES_DIR, 'InAs_nosym.hdf5')
        ])
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_hdf5_file(os.path.join(SAMPLES_DIR, 'InAs_sym_reference.hdf5'))
    models_close(model_res, model_reference)
