#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tempfile
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli

@pytest.mark.parametrize('slice_idx', [
    (3, 1, 2),
    (0, 1, 4, 2, 3, 5, 6, 8, 7, 9, 10, 11, 12)
])
def test_cli_slice(models_equal, slice_idx, sample):
    runner = CliRunner()
    input_file = sample('InAs_nosym.hdf5')
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli,
            [
                'slice',
                '-o', out_file.name,
                '-i', input_file,
            ] + [str(x) for x in slice_idx]
            ,
            catch_exceptions=False
        )
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_hdf5_file(input_file).slice_orbitals(slice_idx=slice_idx)
    models_equal(model_res, model_reference)

@pytest.mark.parametrize('slice_idx', [
    (0, 200),
])
def test_cli_slice_invalid(models_equal, slice_idx, sample):
    runner = CliRunner()
    input_file = sample('InAs_nosym.hdf5')
    with tempfile.NamedTemporaryFile() as out_file:
        with pytest.raises(IndexError):
            runner.invoke(
                cli,
                [
                    'slice',
                    '-o', out_file.name,
                    '-i', input_file,
                ] + [str(x) for x in slice_idx],
                catch_exceptions=False
            )
