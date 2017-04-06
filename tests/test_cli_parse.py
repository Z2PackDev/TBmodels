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

@pytest.mark.parametrize('prefix', ['silicon', 'bi'])
def test_cli_parse(models_equal, prefix):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(cli, ['parse', '-o', out_file.name, '-f', SAMPLES_DIR, '-p', prefix], catch_exceptions=False)
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_wannier_folder(folder=SAMPLES_DIR, prefix=prefix)
    models_equal(model_res, model_reference)
