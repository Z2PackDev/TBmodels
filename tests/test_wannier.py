#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST

from os.path import join

import pytest

import tbmodels
import numpy as np

from parameters import SAMPLES_DIR

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]

@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat', 'silicon_hr.dat'])
def test_wannier_hr_only(compare_data, hr_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)

@pytest.mark.parametrize('hr_name, wsvec_name', [
    ('silicon_hr.dat', 'silicon_wsvec.dat')
])
def test_wannier_hr_wsvec(compare_data, hr_name, wsvec_name):
    model = tbmodels.Model.from_wannier_files(
        hr_file=join(SAMPLES_DIR, hr_name),
        wsvec_file=join(SAMPLES_DIR, wsvec_name)
    )
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)

@pytest.mark.parametrize('hr_name, wsvec_name, xyz_name', [
    ('silicon_hr.dat', 'silicon_wsvec.dat', 'silicon_centres.xyz')
])
def test_wannier_hr_wsvec_xyz(compare_data, hr_name, wsvec_name, xyz_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    wsvec_file = join(SAMPLES_DIR, wsvec_name)
    xyz_file = join(SAMPLES_DIR, xyz_name)
    model = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file,
        xyz_file=xyz_file
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file
    )
    H_list = np.array([model.hamilton(k) for k in kpt])
    H_list2 = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)
    assert np.isclose(H_list, H_list2).all()

    assert np.isclose(model.pos, np.array([
        (-0.46075440, -0.46071138, -0.46076716),
        (-0.46074283,  0.46072157,  0.46071793),
        ( 0.46070307, -0.46076048,  0.46068558),
        ( 0.46070418,  0.46072373, -0.46076362),
        ( 1.81012778,  1.81011207,  1.81011265),
        ( 1.81009687,  0.88866222,  0.88861715),
        ( 0.88863982,  1.81013970,  0.88865990),
        ( 0.88864252,  0.88865189,  1.81009014)
    ]) % 1).all()
@pytest.mark.parametrize('hr_name, wsvec_name, xyz_name, win_name', [
    ('silicon_hr.dat', 'silicon_wsvec.dat', 'silicon_centres.xyz', 'silicon.win')
])
def test_wannier_all(compare_data, hr_name, wsvec_name, xyz_name, win_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    wsvec_file = join(SAMPLES_DIR, wsvec_name)
    xyz_file = join(SAMPLES_DIR, xyz_name)
    win_file = join(SAMPLES_DIR, win_name)
    model = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file,
        xyz_file=xyz_file,
        win_file=win_file
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=hr_file,
        wsvec_file=wsvec_file,
        win_file=win_file
    )
    H_list = np.array([model.hamilton(k) for k in kpt])
    H_list2 = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)
    assert np.isclose(H_list, H_list2).all()

    # check positions
    assert np.isclose(model.pos, np.array([
        (-0.46075440, -0.46071138, -0.46076716),
        (-0.46074283,  0.46072157,  0.46071793),
        ( 0.46070307, -0.46076048,  0.46068558),
        ( 0.46070418,  0.46072373, -0.46076362),
        ( 1.81012778,  1.81011207,  1.81011265),
        ( 1.81009687,  0.88866222,  0.88861715),
        ( 0.88863982,  1.81013970,  0.88865990),
        ( 0.88864252,  0.88865189,  1.81009014)
    ]) % 1).all()

    # check unit cell
    assert np.isclose(model.uc, np.array([
        [-2.6988, 0.0000, 2.6988],
        [0.0000, 2.6988, 2.6988],
        [-2.6988, 2.6988, 0.0000]
    ])).all()

    # check reciprocal lattice
    assert np.isclose(model.reciprocal_lattice, np.array([
        [-1.164070, -1.164070,  1.164070],
        [1.164070 ,  1.164070 ,  1.164070],
        [-1.164070,  1.164070, -1.164070],
    ])).all()

@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat'])
def test_wannier_hr_equal(models_equal, hr_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, occ=28)
    model2 = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    models_equal(model1, model2)

@pytest.mark.parametrize('hr_name', ['wannier90_inconsistent.dat', 'wannier90_inconsistent_v2.dat'])
def test_inconsistent(hr_name):
    with pytest.raises(ValueError):
        model = tbmodels.Model.from_wannier_files(hr_file=join(SAMPLES_DIR, hr_name))


def test_emptylines():
    """test whether the input file with some random empty lines is correctly parsed"""
    model1 = tbmodels.Model.from_wannier_files(hr_file=join(SAMPLES_DIR, 'wannier90_hr.dat'))
    model2 = tbmodels.Model.from_wannier_files(hr_file=join(SAMPLES_DIR, 'wannier90_hr_v2.dat'))
    hop1 = model1.hop
    hop2 = model2.hop
    for k in hop1.keys() | hop2.keys():
        assert (np.array(hop1[k]) == np.array(hop2[k])).all()

def test_error():
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(hr_file=join(SAMPLES_DIR, 'hr_hamilton.dat'), occ=28, pos=[[1., 1., 1.]])
