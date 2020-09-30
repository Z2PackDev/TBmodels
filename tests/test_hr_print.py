#!/usr/bin/env python

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for writing files to *_hr.dat format."""

import tempfile

import pytest

from parameters import T_VALUES
import tbmodels


@pytest.mark.parametrize("t", T_VALUES)
def test_hr_print(t, get_model, compare_equal):
    """Regression test for converting a model to *_hr.dat format."""
    model = get_model(*t)
    compare_equal(model.to_hr().splitlines()[1:])  # timestamp in first line isn't equal


@pytest.mark.parametrize("hr_name", ["hr_hamilton.dat"])
def test_consistency(hr_name, sample):
    """
    Check that the result of loading a *_hr.dat file and converting it
    back to that format creates a result that is consistent with the
    original file.
    """
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28, sparse=True)
    lines_new = model.to_hr().split("\n")
    with open(hr_file) as f:
        lines_old = [line.rstrip(" \r\n") for line in f.readlines()]
    assert len(lines_new) == len(lines_old)
    for l_new, l_old in zip(lines_new[1:], lines_old[1:]):
        assert l_new.replace("-0.00000000000000", " 0.00000000000000") == l_old.replace(
            "-0.00000000000000", " 0.00000000000000"
        )


@pytest.mark.parametrize("hr_name", ["hr_hamilton.dat"])
def test_consistency_file(hr_name, models_equal, sparse, sample):
    """
    Check that a model loaded directly from a *_hr.dat file is equal
    to a model after a save / load round-trip to the same format.
    """
    hr_file = sample(hr_name)
    model1 = tbmodels.Model.from_wannier_files(hr_file=hr_file, sparse=sparse)
    with tempfile.NamedTemporaryFile() as tmpf:
        model1.to_hr_file(tmpf.name)
        model2 = tbmodels.Model.from_wannier_files(hr_file=tmpf.name, sparse=sparse)
    models_equal(model1, model2)


@pytest.mark.parametrize("hr_name", ["hr_hamilton.dat", "hr_hamilton_full.dat"])
def test_consistency_no_hcutoff(hr_name, sample):
    """
    Check that the result of loading a *_hr.dat file and converting it
    back to that format creates a result that is consistent with the
    original file. Loading is performed without h cutoff (with negative
    value).
    """
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_wannier_files(
        hr_file=hr_file, occ=28, h_cutoff=-1, sparse=True
    )
    lines_new = model.to_hr().split("\n")
    with open(hr_file) as f:
        lines_old = [line.rstrip(" \r\n") for line in f.readlines()]
    assert len(lines_new) == len(lines_old)
    for l_new, l_old in zip(lines_new[1:], lines_old[1:]):
        assert l_new.replace("-0.00000000000000", " 0.00000000000000") == l_old.replace(
            "-0.00000000000000", " 0.00000000000000"
        )


def test_invalid_empty():
    """
    Check that trying to convert an incomplete tight-binding model to
    hr format raises an error.
    """
    model = tbmodels.Model(size=2, dim=3)
    with pytest.raises(ValueError):
        model.to_hr()
