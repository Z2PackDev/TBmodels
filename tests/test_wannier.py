#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test loading a tight-binding model from Wannier90 files / folders.
"""

import pytest
import numpy as np

import tbmodels
from tbmodels.exceptions import TbmodelsException

from parameters import KPT


@pytest.mark.parametrize(
    "hr_name",
    ["hr_hamilton.dat", "wannier90_hr.dat", "wannier90_hr_v2.dat", "silicon_hr.dat"],
)
def test_wannier_hr_only(compare_isclose, hr_name, sample):
    """
    Test loading a model from the *_hr.dat file only.
    """
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    hamiltonian_list = np.array([model.hamilton(k) for k in KPT])

    compare_isclose(hamiltonian_list)


@pytest.mark.parametrize(
    "hr_name, wsvec_name",
    [("silicon_hr.dat", "silicon_wsvec.dat"), ("bi_hr.dat", "bi_wsvec.dat")],
)
def test_wannier_hr_wsvec(compare_isclose, hr_name, wsvec_name, sample):
    """
    Test loading a tight-binding model from *_hr.dat and *_wsvec.dat files.
    """
    model = tbmodels.Model.from_wannier_files(
        hr_file=sample(hr_name), wsvec_file=sample(wsvec_name)
    )
    hamiltonian_list = np.array([model.hamilton(k) for k in KPT])

    compare_isclose(hamiltonian_list)


@pytest.mark.parametrize(
    "hr_name, wsvec_name, xyz_name",
    [
        ("silicon_hr.dat", "silicon_wsvec.dat", "silicon_centres.xyz"),
        ("bi_hr.dat", "bi_wsvec.dat", "bi_centres.xyz"),
    ],
)
def test_wannier_hr_wsvec_xyz(hr_name, wsvec_name, xyz_name, sample):
    """
    Check that trying to load positions from .xyz raises an error if no
    unit cell is specified.
    """
    hr_file = sample(hr_name)
    wsvec_file = sample(wsvec_name)
    xyz_file = sample(xyz_name)
    # cannot determine reduced pos if uc is not given
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(
            hr_file=hr_file,
            wsvec_file=wsvec_file,
            xyz_file=xyz_file,
            distance_ratio_threshold=1.0,
        )


@pytest.mark.parametrize(
    "hr_name, wsvec_name, xyz_name, win_name, pos, uc, reciprocal_lattice, pos_kind",
    [
        (  # yapf: disable
            "silicon_hr.dat",
            "silicon_wsvec.dat",
            "silicon_centres.xyz",
            "silicon.win",
            np.array(
                [
                    [0.08535249, -0.25608288, 0.08537316],
                    [0.08536001, 0.08535213, 0.08536136],
                    [0.08536071, 0.08533944, -0.25606735],
                    [-0.25607521, 0.08534613, 0.08536816],
                    [-0.33535779, 1.00606797, -0.335358],
                    [-0.33536052, 0.66462432, -0.33534382],
                    [-0.33535638, 0.66463603, 0.00608418],
                    [0.00607598, 0.66462586, -0.33534919],
                ]
            ),
            np.array(
                [
                    [-2.6988, 0.0000, 2.6988],
                    [0.0000, 2.6988, 2.6988],
                    [-2.6988, 2.6988, 0.0000],
                ]
            ),
            np.array(
                [
                    [-1.164070, -1.164070, 1.164070],
                    [1.164070, 1.164070, 1.164070],
                    [-1.164070, 1.164070, -1.164070],
                ]
            ),
            "wannier",
        ),
        (
            "bi_hr.dat",
            "bi_wsvec.dat",
            "bi_centres.xyz",
            "bi.win",
            np.array(
                [
                    [0.08775382, -0.00245227, -0.05753982],
                    [-0.05495596, 0.14636127, -0.0485512],
                    [-0.06553189, -0.0240915, 0.12595038],
                    [0.08499069, 0.09910585, 0.00461952],
                    [0.07323644, -0.0895939, -0.17143996],
                    [-0.03101997, -0.0348563, 0.01921201],
                    [-0.05896055, -0.00482994, -0.10247473],
                    [0.04122294, -0.01290763, -0.1097073],
                    [0.21852895, -0.16835099, -0.12195773],
                    [-0.02580571, -0.02781471, 0.09245878],
                ]
            ),
            np.array(
                [
                    [2.272990, -1.312311, 3.953982],
                    [0.000000, 2.624622, 3.953982],
                    [-2.272990, -1.312311, 3.953982],
                ]
            ),
            np.array(
                [
                    [1.382141, -0.797980, 0.529693],
                    [0.000000, 1.595959, 0.529693],
                    [-1.382141, -0.797980, 0.529693],
                ]
            ),
            "wannier",
        ),
        (
            "bi_hr.dat",
            "bi_wsvec.dat",
            "bi_centres.xyz",
            "bi.win",
            np.array(
                [
                    [0.76611, 0.76611, 0.76611],
                    [0.76611, 0.76611, 0.76611],
                    [0.76611, 0.76611, 0.76611],
                    [0.23389, 0.23389, 0.23389],
                    [0.76611, 0.76611, 0.76611],
                    [0.76611, 0.76611, 0.76611],
                    [0.76611, 0.76611, 0.76611],
                    [0.23389, 0.23389, 0.23389],
                    [0.76611, 0.76611, 0.76611],
                    [0.76611, 0.76611, 0.76611],
                ]
            ),
            np.array(
                [
                    [2.272990, -1.312311, 3.953982],
                    [0.000000, 2.624622, 3.953982],
                    [-2.272990, -1.312311, 3.953982],
                ]
            ),
            np.array(
                [
                    [1.382141, -0.797980, 0.529693],
                    [0.000000, 1.595959, 0.529693],
                    [-1.382141, -0.797980, 0.529693],
                ]
            ),
            "nearest_atom",
        ),
    ],
)  # pylint: disable=too-many-arguments
def test_wannier_all(
    compare_isclose,
    hr_name,
    wsvec_name,
    xyz_name,
    win_name,
    pos,
    uc,
    reciprocal_lattice,
    sample,
    pos_kind,
):
    """
    Test loading tight-binding models from all Wannier files.
    """
    hr_file = sample(hr_name)
    wsvec_file = sample(wsvec_name)
    xyz_file = sample(xyz_name)
    win_file = sample(win_name)
    model = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file,
        xyz_file=xyz_file,
        win_file=win_file,
        pos_kind=pos_kind,
        distance_ratio_threshold=1.0,
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file,
        win_file=win_file,
        distance_ratio_threshold=1.0,
    )
    hamiltonian_list = np.array([model.hamilton(k) for k in KPT])

    compare_isclose(hamiltonian_list)

    # TODO: Improve test to remove this ugly hack: The 'silicon' .xyz file
    # has positions outside the UC, so the hoppings are mapped in such a
    # way that the two models are not equal.
    if not hr_name.startswith("silicon"):
        hamiltonian_list_2 = np.array([model2.hamilton(k) for k in KPT])
        assert np.allclose(hamiltonian_list, hamiltonian_list_2)
    assert np.allclose(model.pos, pos % 1)
    assert np.allclose(model.uc, uc)
    assert np.allclose(model.reciprocal_lattice, reciprocal_lattice)


@pytest.mark.parametrize(
    "hr_name", ["wannier90_inconsistent.dat", "wannier90_inconsistent_v2.dat"]
)
def test_inconsistent(hr_name, sample):
    """
    Check that trying to load inconsistent *_hr.dat files raises an error.
    """
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(hr_file=sample(hr_name))


def test_emptylines(sample):
    """test whether the input file with some random empty lines is correctly parsed"""
    model1 = tbmodels.Model.from_wannier_files(hr_file=sample("wannier90_hr.dat"))
    model2 = tbmodels.Model.from_wannier_files(hr_file=sample("wannier90_hr_v2.dat"))
    hop1 = model1.hop
    hop2 = model2.hop
    for k in hop1.keys() | hop2.keys():
        assert (np.array(hop1[k]) == np.array(hop2[k])).all()


def test_error(sample):
    """Check that passing the wrong number of positions raises an error."""
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(
            hr_file=sample("hr_hamilton.dat"), occ=28, pos=[[1.0, 1.0, 1.0]]
        )


def test_win_and_uc(sample):
    """Check that passing both the unit cell and win raises an error."""
    with pytest.raises(ValueError) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("silicon_hr.dat"),
            win_file=sample("silicon.win"),
            uc=np.eye(3),
        )
    assert "Ambiguous unit cell" in str(excinfo.value)


def test_xyz_and_pos(sample):
    """Check that passing both the positions and .xyz raises an error."""
    with pytest.raises(ValueError) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("silicon_hr.dat"),
            win_file=sample("silicon.win"),
            xyz_file=sample("silicon_centres.xyz"),
            pos=[(0, 0, 0)] * 10,
        )
    assert "Ambiguous orbital positions" in str(excinfo.value)


def test_invalid_distance_ratio_threshold(sample):  # pylint: disable=invalid-name
    """
    Check that passing an invalid 'distance_ratio_threshol' value raises
    an error.
    """
    with pytest.raises(ValueError) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("silicon_hr.dat"),
            win_file=sample("silicon.win"),
            xyz_file=sample("silicon_centres.xyz"),
            pos_kind="nearest_atom",
            distance_ratio_threshold=0.5,
        )
    assert "Invalid value for 'distance_ratio_threshold'" in str(excinfo.value)


def test_invalid_pos_kind(sample):
    """
    Check that passing an invalid 'pos_kind' value raises an error.
    """
    with pytest.raises(ValueError) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("silicon_hr.dat"),
            win_file=sample("silicon.win"),
            xyz_file=sample("silicon_centres.xyz"),
            pos_kind="whatever",
        )
    assert "Invalid value 'whatever' for 'pos_kind'" in str(excinfo.value)


def test_wsvec_blocks_missing(sample):
    """
    Check that a wsvec file with entire missing entries raises KeyError.

    In this case, the individual blocks in the wsvec file are complete,
    but entire blocks are missing.
    """
    with pytest.raises(KeyError):
        tbmodels.Model.from_wannier_files(
            hr_file=sample("bi_hr.dat"),
            wsvec_file=sample("bi_wsvec_blocks_missing.dat"),
            xyz_file=sample("bi_centres.xyz"),
            win_file=sample("bi.win"),
        )


def test_wsvec_blocks_incomplete(sample):
    """
    Check that a wsvec file with incomplete blocks raises an error.
    """
    with pytest.raises(TbmodelsException) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("bi_hr.dat"),
            wsvec_file=sample("bi_wsvec_blocks_incomplete.dat"),
            xyz_file=sample("bi_centres.xyz"),
            win_file=sample("bi.win"),
        )
    assert "Incomplete wsvec iterator." in str(excinfo.value)


def test_wsvec_empty(sample):
    """
    Check that an empty wsvec file raises an error.
    """
    with pytest.raises(TbmodelsException) as excinfo:
        tbmodels.Model.from_wannier_files(
            hr_file=sample("bi_hr.dat"),
            wsvec_file=sample("bi_wsvec_empty.dat"),
            xyz_file=sample("bi_centres.xyz"),
            win_file=sample("bi.win"),
        )
    assert "The 'wsvec' iterator is empty." in str(excinfo.value)


def test_length_unit_modes(sample, models_equal):
    """
    Check that specifying the length unit in .win explicitly or in the
    unit cell block is equivalent.
    """
    model1 = tbmodels.Model.from_wannier_files(
        hr_file=sample("bi_hr.dat"),
        win_file=sample("bi.win"),
        xyz_file=sample("bi_centres.xyz"),
        wsvec_file=sample("bi_wsvec.dat"),
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=sample("bi_hr.dat"),
        win_file=sample("bi_equivalent.win"),
        xyz_file=sample("bi_centres.xyz"),
        wsvec_file=sample("bi_wsvec.dat"),
    )
    assert models_equal(model1, model2)
