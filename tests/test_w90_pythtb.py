#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os

import pytest
import numpy as np
import pythtb as pt
import tbmodels as tb

from parameters import SAMPLES_DIR

def test_si_pythtb():
    pt_model = pt.w90(SAMPLES_DIR, 'silicon').model()
    tb_model = tb.Model.from_wannier_files(
        hr_file=os.path.join(SAMPLES_DIR, 'silicon_hr.dat'),
        win_file=os.path.join(SAMPLES_DIR, 'silicon.win'),
        # This might be needed if pythtb supports wsvec.dat
        # wsvec_file=os.path.join(SAMPLES_DIR, 'silicon_wsvec.dat'),
        xyz_file=os.path.join(SAMPLES_DIR, 'silicon_centres.xyz')
    )

    assert np.allclose(pt_model._gen_ham([0, 0, 0]), tb_model.hamilton([0, 0, 0]))
    assert np.allclose(pt_model._gen_ham([0, 0, 0]), tb_model.hamilton([0, 0, 0], convention=1))

    k = (0.123412512, 0.6234615, 0.72435235)
    assert np.allclose(pt_model._gen_ham(k), tb_model.hamilton(k, convention=1))
    assert np.allclose(pt_model.get_lat(), tb_model.uc)
    assert np.allclose(pt_model.get_orb() % 1, tb_model.pos)
