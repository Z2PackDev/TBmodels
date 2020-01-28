#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test helper function converting hopping matrices to a list of hoppings.
"""

import numpy as np

import tbmodels


def test_consistency(get_model_clean, models_equal, sparse):
    """
    Check that converting all hopping matrices to individual hoppings
    and using these to re-construct the model yields the same model.
    """
    model1 = get_model_clean(0.1, 0.2, sparse=sparse)

    hoppings = []
    for recip_lattice_vector, hop_mat in model1.hop.items():
        hoppings.extend(tbmodels.helpers.matrix_to_hop(np.array(hop_mat), R=recip_lattice_vector))

    model2 = tbmodels.Model.from_hop_list(
        size=2,
        hop_list=hoppings,
        contains_cc=False,
        occ=1,
        pos=((0., ) * 3, (0.5, 0.5, 0.)),
        sparse=sparse
    )
    models_equal(model1, model2)
