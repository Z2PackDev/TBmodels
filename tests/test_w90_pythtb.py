#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pythtb as pt
import tbmodels as tb

@pytest.mark.parametrize('prefix', ['silicon']) # bi takes too long
def test_consistency_pythtb(prefix, sample):
    pt_model = pt.w90(sample(''), prefix).model()
    tb_model = tb.Model.from_wannier_files(
        hr_file=sample(prefix + '_hr.dat'),
        win_file=sample(prefix + '.win'),
        # This might be needed if pythtb supports wsvec.dat
        # wsvec_file=sample('silicon_wsvec.dat'),
        xyz_file=sample(prefix + '_centres.xyz')
    )

    assert np.allclose(pt_model._gen_ham([0, 0, 0]), tb_model.hamilton([0, 0, 0]))
    assert np.allclose(pt_model._gen_ham([0, 0, 0]), tb_model.hamilton([0, 0, 0], convention=1))

    k = (0.123412512, 0.6234615, 0.72435235)
    assert np.allclose(pt_model._gen_ham(k), tb_model.hamilton(k, convention=1))
    assert np.allclose(pt_model.get_lat(), tb_model.uc)
    assert np.allclose(pt_model.get_orb() % 1, tb_model.pos)
