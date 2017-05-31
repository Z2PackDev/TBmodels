#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tempfile
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli

@pytest.mark.parametrize('prefix', ['silicon', 'bi'])
def test_cli_parse(models_equal, prefix, sample):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(cli, ['parse', '-o', out_file.name, '-f', sample(''), '-p', prefix], catch_exceptions=False)
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_wannier_folder(folder=sample(''), prefix=prefix)
    models_equal(model_res, model_reference)
