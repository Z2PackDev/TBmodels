#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import tbmodels
import numpy as np

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]


@pytest.mark.parametrize(
    'hr_name', [
        'hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat',
        'silicon_hr.dat'
    ]
)
def test_wannier_hr_only(compare_isclose, hr_name, sample):
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_isclose(H_list)


@pytest.mark.parametrize(
    'hr_name, wsvec_name', [('silicon_hr.dat', 'silicon_wsvec.dat'),
                            ('bi_hr.dat', 'bi_wsvec.dat')]
)
def test_wannier_hr_wsvec(compare_isclose, hr_name, wsvec_name, sample):
    model = tbmodels.Model.from_wannier_files(
        hr_file=sample(hr_name), wsvec_file=sample(wsvec_name)
    )
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_isclose(H_list)


@pytest.mark.parametrize(
    'hr_name, wsvec_name, xyz_name',
    [('silicon_hr.dat', 'silicon_wsvec.dat', 'silicon_centres.xyz'),
     ('bi_hr.dat', 'bi_wsvec.dat', 'bi_centres.xyz')]
)
def test_wannier_hr_wsvec_xyz(hr_name, wsvec_name, xyz_name, sample):
    hr_file = sample(hr_name)
    wsvec_file = sample(wsvec_name)
    xyz_file = sample(xyz_name)
    # cannot determine reduced pos if uc is not given
    with pytest.raises(ValueError):
        model = tbmodels.Model.from_wannier_files(
            hr_file=hr_file, wsvec_file=wsvec_file, xyz_file=xyz_file
        )


@pytest.mark.parametrize(
    'hr_name, wsvec_name, xyz_name, win_name, pos, uc, reciprocal_lattice, pos_kind',
    [( # yapf: disable
        'silicon_hr.dat',
        'silicon_wsvec.dat',
        'silicon_centres.xyz',
        'silicon.win',
        np.array([
            [0.08535249, -0.25608288, 0.08537316],
            [0.08536001, 0.08535213, 0.08536136],
            [0.08536071, 0.08533944, -0.25606735],
            [-0.25607521, 0.08534613, 0.08536816],
            [-0.33535779, 1.00606797, -0.335358],
            [-0.33536052, 0.66462432, -0.33534382],
            [-0.33535638, 0.66463603, 0.00608418],
            [0.00607598, 0.66462586, -0.33534919]
        ]),
        np.array([
            [-2.6988, 0.0000, 2.6988],
            [0.0000, 2.6988, 2.6988],
            [-2.6988, 2.6988, 0.0000]
        ]),
        np.array([
            [-1.164070, -1.164070, 1.164070],
            [1.164070, 1.164070, 1.164070],
            [-1.164070, 1.164070, -1.164070],
        ]),
        'wannier',
    ), (
        'bi_hr.dat',
        'bi_wsvec.dat',
        'bi_centres.xyz',
        'bi.win',
        np.array([
            [0.08775382, -0.00245227, -0.05753982],
            [-0.05495596, 0.14636127, -0.0485512],
            [-0.06553189, -0.0240915, 0.12595038],
            [0.08499069, 0.09910585, 0.00461952],
            [0.07323644, -0.0895939, -0.17143996],
            [-0.03101997, -0.0348563, 0.01921201],
            [-0.05896055, -0.00482994, -0.10247473],
            [0.04122294, -0.01290763, -0.1097073],
            [0.21852895, -0.16835099, -0.12195773],
            [-0.02580571, -0.02781471, 0.09245878]
        ]),
        np.array([[2.272990, -1.312311, 3.953982], [
            0.000000, 2.624622, 3.953982
        ], [-2.272990, -1.312311, 3.953982]]),
        np.array([
            [1.382141, -0.797980, 0.529693],
            [0.000000, 1.595959, 0.529693],
            [-1.382141, -0.797980, 0.529693],
        ]),
        'wannier',
    ), (
        'bi_hr.dat',
        'bi_wsvec.dat',
        'bi_centres.xyz',
        'bi.win',
        np.array([
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
            [0.23389, 0.23389, 0.23389],
        ]),
        np.array([[2.272990, -1.312311, 3.953982], [
            0.000000, 2.624622, 3.953982
        ], [-2.272990, -1.312311, 3.953982]]),
        np.array([
            [1.382141, -0.797980, 0.529693],
            [0.000000, 1.595959, 0.529693],
            [-1.382141, -0.797980, 0.529693],
        ]),
        'nearest_atom',
    )]
)
def test_wannier_all(
    compare_isclose, hr_name, wsvec_name, xyz_name, win_name, pos, uc,
    reciprocal_lattice, sample, pos_kind
):
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
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=hr_file, wsvec_file=wsvec_file, win_file=win_file
    )
    H_list = np.array([model.hamilton(k) for k in kpt])
    H_list2 = np.array([model.hamilton(k) for k in kpt])

    compare_isclose(H_list)
    assert np.isclose(H_list, H_list2).all()
    assert np.allclose(model.pos, pos % 1)
    assert np.allclose(model.uc, uc)
    assert np.allclose(model.reciprocal_lattice, reciprocal_lattice)


@pytest.mark.parametrize(
    'hr_name', ['hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat']
)
def test_wannier_hr_equal(models_equal, hr_name, sample):
    hr_file = sample(hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, occ=28)
    model2 = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    models_equal(model1, model2)


@pytest.mark.parametrize(
    'hr_name', ['wannier90_inconsistent.dat', 'wannier90_inconsistent_v2.dat']
)
def test_inconsistent(hr_name, sample):
    with pytest.raises(ValueError):
        model = tbmodels.Model.from_wannier_files(hr_file=sample(hr_name))


def test_emptylines(sample):
    """test whether the input file with some random empty lines is correctly parsed"""
    model1 = tbmodels.Model.from_wannier_files(
        hr_file=sample('wannier90_hr.dat')
    )
    model2 = tbmodels.Model.from_wannier_files(
        hr_file=sample('wannier90_hr_v2.dat')
    )
    hop1 = model1.hop
    hop2 = model2.hop
    for k in hop1.keys() | hop2.keys():
        assert (np.array(hop1[k]) == np.array(hop2[k])).all()


def test_error(sample):
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(
            hr_file=sample('hr_hamilton.dat'), occ=28, pos=[[1., 1., 1.]]
        )
