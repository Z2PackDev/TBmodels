#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    trs.py

from common import *

import numpy as np

class EmFielTestCase(SimpleModelTestCase):
    #---------------- e-field (scalar potential)----------------------------#
    def test_no_e(self):
        self.createH(0.2, 0.3, uc=np.identity(3))
        em_model_0 = self.model.em_field()
        em_model_1 = self.model.em_field(lambda r: 0.)
        self.assertFullAlmostEqual(em_model_0.hamilton([0.1, 0.2, 0.3]), em_model_1.hamilton([0.1, 0.2, 0.3]))

    def test_e_0(self):
        self.createH(0.2, 0.3, uc=np.identity(3))
        em_model = self.model.em_field(lambda r: 0.1 * r[0] + 0.2 * r[1]**2 - 0.7 * r[2], lambda r: [0., 0., 0.])

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))

    def test_e_1(self):
        self.createH(0.2, 0.3, uc=np.array([[0.1, 4., 0.2], [0, 1, 2], [0, 0, 0.3]]))
        em_model = self.model.em_field(lambda r: 0.1 * r[0] + 0.1 * r[1]**2 - 0.7 * r[2], lambda r: [0., 0., 0.])

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))

    def test_e_2(self):
        self.createH(0.2, 0.3, uc=np.array([[0.1, 4., 0.2], [0, 1, 2], [0, 0, 0.3]]))
        em_model = self.model.em_field(lambda r: 0.1 * r[0] + 0.1 * r[1]**2 - 0.7 * r[2], lambda r: [0., 0., 0.], prefactor_scalar=3.)

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))

    #---------------- m-field (vector potential) ---------------------------#
    def test_no_m(self):
        self.createH(0.2, 0.3, uc=np.identity(3))
        em_model_0 = self.model.em_field()
        em_model_1 = self.model.em_field(vec_pot=lambda r: [0., 0., 0.])
        self.assertFullAlmostEqual(em_model_0.hamilton([0.1, 0.2, 0.3]), em_model_1.hamilton([0.1, 0.2, 0.3]))

    def test_m_0(self):
        self.createH(0.2, 0.3, uc=np.identity(3))
        em_model = self.model.em_field(vec_pot=lambda r: [0.1 * r[0] + r[1]**2, 0.1 * r[1]**2, -0.7 * r[2] - r[0]])

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))

    def test_m_1(self):
        self.createH(0.2, 0.3, uc=np.array([[0.1, 4., 0.2], [0, 1, 2], [0, 0, 0.3]]))
        em_model = self.model.em_field(vec_pot=lambda r: [0.1 * r[0] + r[1]**2, 0.1 * r[1]**2 - 0.7 * r[2], -r[0]])

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))

    def test_m_2(self):
        self.createH(0.2, 0.3, uc=np.array([[0.1, 4., 0.2], [0, 1, 2], [0, 0, 0.3]]))
        em_model = self.model.em_field(vec_pot=lambda r: [0.1 * r[0] + r[1]**2, 0.1 * r[1]**2, -0.7 * r[2] - r[0]], prefactor_vec=3.)

        res = in_place_replace(em_model.hamilton([0.1, 0.2, 0.3]))
        self.assertFullAlmostEqual(res, em_model.hamilton([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
