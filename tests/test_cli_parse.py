#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for the 'parse' CLI command."""

import tempfile

import pytest
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli


@pytest.mark.parametrize('pos_kind', ['wannier', 'nearest_atom'])
@pytest.mark.parametrize('prefix', ['silicon', 'bi'])
def test_cli_parse(
    models_equal, prefix, sample, pos_kind, cli_sparsity_arguments, modify_reference_model_sparsity
):
    """Test the 'parse' command with different 'prefix' and 'pos_kind'."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli,
            ['parse', '-o', out_file.name, '-f',
             sample(''), '-p', prefix, '--pos-kind', pos_kind] + cli_sparsity_arguments,
            catch_exceptions=False
        )
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_wannier_folder(
        folder=sample(''), prefix=prefix, pos_kind=pos_kind
    )
    modify_reference_model_sparsity(model_reference)
    models_equal(model_res, model_reference)
