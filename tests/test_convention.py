#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2015-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import pythtb as pt
import tbmodels as tb


def test_compare_pythtb():
    pt_model = pt.tb_model(1, 1, lat=[[1]], orb=[[0], [0.2]])
    tb_model = tb.Model(dim=1, pos=[[0], [0.2]], uc=[[1]])

    pt_model.set_hop(3j, 0, 1, [1])
    tb_model.add_hop(3j, 0, 1, [1])

    assert np.isclose(pt_model._gen_ham([0]), tb_model.hamilton([0])).all()
    assert np.isclose(pt_model._gen_ham([0]), tb_model.hamilton([0], convention=1)).all()

    assert np.isclose(pt_model._gen_ham([1]), tb_model.hamilton([1], convention=1)).all()
    assert np.isclose(pt_model._gen_ham([0.2]), tb_model.hamilton(0.2, convention=1)).all()
