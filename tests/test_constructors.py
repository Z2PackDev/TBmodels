#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test the Model constructors.
"""

import itertools

import pytest
import numpy as np

import tbmodels


def test_on_site_too_long(get_model):
    """
    Check that error is raised when the on_site list is too long.
    """
    with pytest.raises(ValueError):
        get_model(0.1, 0.2, on_site=[1, 2, 3])


def test_no_size_given(get_model, models_equal):
    """
    Check that the Model can be created without explicit size,
    """
    model1 = get_model(0.1, 0.2, size=None)
    model2 = get_model(0.1, 0.2)
    models_equal(model1, model2)


def test_size_unknown(get_model):
    """
    Check that an error is raised when the size can not be inferred.
    """
    with pytest.raises(ValueError):
        get_model(0.1, 0.2, size=None, on_site=None, pos=None)


def test_add_on_site(get_model, models_equal):
    """
    Check that adding on-site interaction in the constructor has the
    same as effect as adding it after construction.
    """
    model1 = get_model(0.1, 0.2, on_site=(1, -2))
    model2 = get_model(0.1, 0.2, size=2, on_site=None)
    model2.add_on_site((1, -2))
    models_equal(model1, model2)


def test_invalid_add_on_site(get_model):
    """
    Check that an error is raised when trying to add a list of on-site
    interactions that is too long to an existing model.
    """
    model = get_model(0.1, 0.2)
    with pytest.raises(ValueError):
        model.add_on_site((1, 2, 3))


def test_explicit_dim(get_model, models_equal):
    """
    Check that explicitly setting the dimension does not change the model.
    """
    model1 = get_model(0.1, 0.2, dim=3)
    model2 = get_model(0.1, 0.2)
    models_equal(model1, model2)


def test_no_dim(get_model):
    """Check that an error is raised when the dimension can not be inferred."""
    with pytest.raises(ValueError):
        get_model(0.1, 0.2, pos=None)


def test_pos_outside_uc(get_model, models_equal):
    """Check that positions outside the UC are mapped back inside."""
    model1 = get_model(0.1, 0.2, pos=((0., 0., 0.), (-0.5, -0.5, 0.)))
    model2 = get_model(0.1, 0.2)
    models_equal(model1, model2)


@pytest.mark.parametrize('sparse', [True, False])
def test_from_hop_list(get_model, models_equal, sparse):
    """
    Check the 'from_hop_list' constructor.
    """
    t1 = 0.1
    t2 = 0.2
    hoppings = []
    for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1], [0])):
        hoppings.append([t1 * phase, 0, 1, R])

    for R in ((r[0], r[1], 0) for r in itertools.permutations([0, 1])):
        hoppings.append([t2, 0, 0, R])
        hoppings.append([-t2, 1, 1, R])
    model1 = tbmodels.Model.from_hop_list(
        hop_list=hoppings,
        contains_cc=False,
        on_site=(1, -1),
        occ=1,
        pos=((0., ) * 3, (0.5, 0.5, 0.)),
        sparse=sparse
    )
    model2 = get_model(t1, t2, sparse=sparse)
    models_equal(model1, model2)


@pytest.mark.parametrize('sparse', [True, False])
def test_from_hop_list_with_cc(get_model, models_close, sparse):
    """
    Check the 'from_hop_list' constructor, where complex conjugate terms
    are included in the list.
    """
    t1 = 0.1
    t2 = 0.2
    hoppings = []
    for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1], [0])):
        hoppings.append([t1 * phase, 0, 1, R])

    for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1], [0])):
        hoppings.append([np.conjugate(t1 * phase), 1, 0, tuple(-x for x in R)])

    for R in ((r[0], r[1], 0) for r in itertools.permutations([0, 1])):
        hoppings.append([t2, 0, 0, R])
        hoppings.append([t2, 0, 0, tuple(-x for x in R)])
        hoppings.append([-t2, 1, 1, R])
        hoppings.append([-t2, 1, 1, tuple(-x for x in R)])
    model1 = tbmodels.Model.from_hop_list(
        hop_list=hoppings,
        contains_cc=True,
        on_site=(1, -1),
        occ=1,
        pos=((0., ) * 3, (0.5, 0.5, 0.)),
        sparse=sparse
    )
    model2 = get_model(t1, t2, sparse=sparse)
    models_close(model1, model2)


@pytest.mark.parametrize('sparse', [True, False])
def test_pos_outside_uc_with_hoppings(get_model, models_equal, sparse):  # pylint: disable=invalid-name
    """
    Check the 'from_hop_list' constructor with positions outside of the UC.
    """
    t1 = 0.1
    t2 = 0.2
    hoppings = []
    for phase, R in zip([1, -1j, 1j, -1], [(1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]):
        hoppings.append([t1 * phase, 0, 1, R])

    for R in ((r[0], r[1], 0) for r in itertools.permutations([0, 1])):
        hoppings.append([t2, 0, 0, R])
        hoppings.append([-t2, 1, 1, R])
    model1 = tbmodels.Model.from_hop_list(
        hop_list=hoppings,
        contains_cc=False,
        on_site=(1, -1),
        occ=1,
        pos=((0., ) * 3, (-0.5, -0.5, 0.)),
        sparse=sparse
    )
    model2 = get_model(t1, t2, sparse=sparse)
    models_equal(model1, model2)


def test_invalid_hopping_matrix():
    """
    Check that an error is raised when the passed size does not match the
    shape of hopping matrices.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(size=2, hop={(0, 0, 0): np.eye(4)})


def test_non_hermitian_1():
    """
    Check that an error is raised when the given hoppings do not correspond
    to a hermitian Hamiltonian.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(size=2, hop={(0, 0, 0): np.eye(2), (1, 0, 0): np.eye(2)})


def test_non_hermitian_2():
    """
    Check that an error is raised when the given hoppings do not correspond
    to a hermitian Hamiltonian.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(
            size=2, hop={(0, 0, 0): np.eye(2),
                         (1, 0, 0): np.eye(2),
                         (-1, 0, 0): 2 * np.eye(2)}
        )


def test_wrong_key_length():
    """
    Check that an error is raised when the reciprocal lattice vectors
    have inconsistent lengths.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(
            size=2,
            hop={(0, 0, 0): np.eye(2),
                 (1, 0, 0): np.eye(2),
                 (-1, 0, 0, 0): np.eye(2)},
            contains_cc=False
        )


def test_wrong_pos_length():
    """
    Check that an error is raised when the number of positions does not
    match the given size.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(
            size=2,
            hop={(0, 0, 0): np.eye(2),
                 (1, 0, 0): np.eye(2),
                 (-1, 0, 0): np.eye(2)},
            contains_cc=False,
            pos=((0., ) * 3, (0.5, ) * 3, (0.2, ) * 3)
        )


def test_wrong_pos_dim():
    """
    Check that an error is raised when the positions have inconsistent
    dimensions.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(
            size=2,
            hop={(0, 0, 0): np.eye(2),
                 (1, 0, 0): np.eye(2),
                 (-1, 0, 0): np.eye(2)},
            contains_cc=False,
            pos=((0., ) * 3, (0.5, ) * 4)
        )


def test_wrong_uc_shape():
    """
    Check that an error is raised when the unit cell is not square.
    """
    with pytest.raises(ValueError):
        tbmodels.Model(
            size=2,
            hop={(0, 0, 0): np.eye(2),
                 (1, 0, 0): np.eye(2),
                 (-1, 0, 0): np.eye(2)},
            contains_cc=False,
            pos=((0., ) * 3, (0.5, ) * 3),
            uc=np.array([[1, 2], [3, 4], [5, 6]])
        )


def test_hop_list_no_size():
    """
    Check that an error is raised when using 'from_hop_list' and
    the size is not known.
    """
    with pytest.raises(ValueError):
        tbmodels.Model.from_hop_list(hop_list=(1.2, 0, 1, (1, 2, 3)))
