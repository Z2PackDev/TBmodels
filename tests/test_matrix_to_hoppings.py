#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import tbmodels
import numpy as np


def test_consistency(get_model_clean, models_equal, sparse):
    model1 = get_model_clean(0.1, 0.2, sparse=sparse)

    hoppings = []
    for k, v in model1.hop.items():
        hoppings.extend(tbmodels.helpers.matrix_to_hop(np.array(v), R=k))

    model2 = tbmodels.Model.from_hop_list(
        size=2,
        hop_list=hoppings,
        contains_cc=False,
        occ=1,
        pos=((0., ) * 3, (0.5, 0.5, 0.)),
        sparse=sparse
    )
    models_equal(model1, model2)
