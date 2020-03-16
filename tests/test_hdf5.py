#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests saving and loading to HDF5 format."""

import tempfile

import pytest
import numpy as np

import tbmodels

KWARGS = [
    dict(),
    dict(pos=None, dim=3),
    dict(uc=3 * np.eye(3)),
    dict(pos=np.zeros((2, 3)), uc=np.eye(3)),
]


@pytest.mark.parametrize("kwargs", KWARGS)
def test_hdf5_consistency_file(get_model, models_equal, kwargs):
    """
    Test that a tight-binding model remains the same after saving to
    HDF5 format and loading, using the Model methods.
    """
    model1 = get_model(0.1, 0.2, **kwargs)
    with tempfile.NamedTemporaryFile() as tmpf:
        model1.to_hdf5_file(tmpf.name)
        model2 = tbmodels.Model.from_hdf5_file(tmpf.name)
    models_equal(model1, model2)


@pytest.mark.parametrize("kwargs", KWARGS)
def test_hdf5_consistency_freefunc(get_model, models_equal, kwargs):
    """
    Test that a tight-binding model remains the same after saving to
    HDF5 and loading, using the free `io` functions.
    """
    model1 = get_model(0.1, 0.2, **kwargs)
    with tempfile.NamedTemporaryFile() as tmpf:
        tbmodels.io.save(model1, tmpf.name)
        model2 = tbmodels.io.load(tmpf.name)
    models_equal(model1, model2)


@pytest.fixture(params=["InAs_nosym.hdf5"])
def hdf5_sample(sample, request):
    """Fixture which provides the filename of a HDF5 tight-binding model."""
    return sample(request.param)


@pytest.fixture(params=["InAs_nosym_legacy.hdf5"])
def hdf5_sample_legacy(sample, request):
    """
    Fixture which provides the filename of a HDF5 tight-binding model,
    in legacy format.
    """
    return sample(request.param)


def test_hdf5_load_freefunc(hdf5_sample):  # pylint: disable=redefined-outer-name
    """Test that a HDF5 file can be loaded with the `io.load` function."""
    res = tbmodels.io.load(hdf5_sample)
    assert isinstance(res, tbmodels.Model)


def test_hdf5_load_method(hdf5_sample):  # pylint: disable=redefined-outer-name
    """Test that a HDF5 file can be loaded with the Model method."""
    res = tbmodels.Model.from_hdf5_file(hdf5_sample)
    assert isinstance(res, tbmodels.Model)


def test_hdf5_load_freefunc_legacy(
    hdf5_sample_legacy,
):  # pylint: disable=redefined-outer-name
    """Test that a HDF5 file in legacy format can be loaded with the `io.load` function."""
    with pytest.deprecated_call():
        res = tbmodels.io.load(hdf5_sample_legacy)
    assert isinstance(res, tbmodels.Model)


def test_hdf5_load_method_legacy(
    hdf5_sample_legacy,
):  # pylint: disable=redefined-outer-name
    """Test that a HDF5 file in legacy format can be loaded with the Model method."""
    with pytest.deprecated_call():
        res = tbmodels.Model.from_hdf5_file(hdf5_sample_legacy)
    assert isinstance(res, tbmodels.Model)


def test_hdf5_kdotp(kdotp_models_equal):
    """Test that k.p models can be saved / loaded to HDF5."""
    kp_model = tbmodels.kdotp.KdotpModel(
        {(1, 0): [[0.1, 0.2j], [-0.2j, 0.3]], (0, 0): np.eye(2)}
    )
    with tempfile.NamedTemporaryFile() as tmpf:
        tbmodels.io.save(kp_model, tmpf.name)
        model2 = tbmodels.io.load(tmpf.name)
    kdotp_models_equal(kp_model, model2)


def test_generic_legacy_object(sample):
    """Test that a generic object in legacy format can be loaded."""
    filename = sample("legacy_general_object.hdf5")
    with pytest.deprecated_call():
        res = tbmodels.io.load(filename)
    assert res == [2, [3, 4], 2.3]
