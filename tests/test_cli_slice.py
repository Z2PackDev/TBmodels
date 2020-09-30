#!/usr/bin/env python

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests the 'slice' CLI command."""

import tempfile

import pytest
from click.testing import CliRunner

import tbmodels
from tbmodels._cli import cli


@pytest.mark.parametrize(
    "slice_idx", [(3, 1, 2), (0, 1, 4, 2, 3, 5, 6, 8, 7, 9, 10, 11, 12)]
)
def test_cli_slice(
    models_equal,
    slice_idx,
    sample,
    cli_sparsity_arguments,
    cli_verbosity_argument,
    modify_reference_model_sparsity,
):
    """
    Check that using the CLI to slice a tight-binding model produces
    the same result as using the `slice_orbitals` method.
    """
    runner = CliRunner()
    input_file = sample("InAs_nosym.hdf5")
    with tempfile.NamedTemporaryFile() as out_file:
        run = runner.invoke(
            cli,
            [
                "slice",
                "-o",
                out_file.name,
                "-i",
                input_file,
            ]
            + cli_sparsity_arguments
            + cli_verbosity_argument
            + [str(x) for x in slice_idx],
            catch_exceptions=False,
        )
        print(run.output)
        model_res = tbmodels.Model.from_hdf5_file(out_file.name)
    model_reference = tbmodels.Model.from_hdf5_file(input_file).slice_orbitals(
        slice_idx=slice_idx
    )
    modify_reference_model_sparsity(model_reference)
    models_equal(model_res, model_reference)


def test_cli_slice_invalid(sample):
    """
    Check that passing an invalid index to the 'slice' command raises
    an error.
    """
    runner = CliRunner()
    input_file = sample("InAs_nosym.hdf5")
    with tempfile.NamedTemporaryFile() as out_file:
        with pytest.raises(IndexError):
            runner.invoke(
                cli,
                ["slice", "-o", out_file.name, "-i", input_file, "0", "200"],
                catch_exceptions=False,
            )
