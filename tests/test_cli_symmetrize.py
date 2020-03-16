#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the 'symmetrize' CLI command.
"""

import tempfile

from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli


def test_cli_symmetrize(
    models_close,
    sample,
    cli_sparsity_arguments,
    cli_verbosity_argument,
    modify_reference_model_sparsity,
):
    """
    Test the 'symmetrize' command.
    """
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli,
            [
                "symmetrize",
                "-o",
                out_file.name,
                "-s",
                sample("InAs_symmetries.hdf5"),
                "-i",
                sample("InAs_nosym.hdf5"),
            ]
            + cli_sparsity_arguments
            + cli_verbosity_argument,
            catch_exceptions=False,
        )
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_hdf5_file(sample("InAs_sym_reference.hdf5"))
    modify_reference_model_sparsity(model_reference)
    models_close(model_res, model_reference)
