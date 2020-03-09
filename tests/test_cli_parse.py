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
@pytest.mark.parametrize('prefix, distance_ratio_threshold', [('silicon', 2.), ('bi', 1)])
def test_cli_parse(  # pylint: disable=too-many-arguments
    models_equal, prefix, sample, pos_kind, cli_sparsity_arguments, cli_verbosity_argument, modify_reference_model_sparsity,
    distance_ratio_threshold
):
    """Test the 'parse' command with different 'prefix' and 'pos_kind'."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        if distance_ratio_threshold is None:
            distance_ratio_arguments = []
            distance_ratio_kwargs = {}
        else:
            distance_ratio_arguments = ['--distance-ratio-threshold', str(distance_ratio_threshold)]
            distance_ratio_kwargs = {'distance_ratio_threshold': distance_ratio_threshold}
        run = runner.invoke(
            cli,
            ['parse', '-o', out_file.name, '-f',
             sample(''), '-p', prefix, '--pos-kind', pos_kind] + cli_sparsity_arguments +
            cli_verbosity_argument + distance_ratio_arguments,
            catch_exceptions=False
        )
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_wannier_folder(
        folder=sample(''), prefix=prefix, pos_kind=pos_kind, **distance_ratio_kwargs
    )
    modify_reference_model_sparsity(model_reference)
    models_equal(model_res, model_reference)


@pytest.mark.parametrize('prefix', ['silicon', 'bi'])
def test_ambiguous_nearest_atom(prefix, sample):
    """
    Test that the 'parse' command with `pos_kind='nearest_atom' results
    in the expected error for cases where the nearest atom position is
    ambiguous.
    """
    runner = CliRunner(mix_stderr=False)
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli, [
                'parse', '-o', out_file.name, '-f',
                sample(''), '-p', prefix, '--pos-kind', 'nearest_atom'
            ],
            catch_exceptions=False
        )
        assert run.stderr.startswith('Error: [AMBIGUOUS_NEAREST_ATOM_POSITIONS]')
