#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    change_uc.py

from common import *

import numpy as np

class ChangeUcTestCase(CommonTestCase):
    def createH(self, t1, t2, uc=None):
        model = tbmodels.Model(
            on_site=[1, -1, 0],
            pos=[
                [0, 0., 0.],
                [0.5, 0.5, 0.2],
                [0.75, 0.15, 0.6],
            ],
            occ=1,
            uc=uc,
            dim=3
        )
        
        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hop(t1 * phase, 0, 1, G)

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hop(t2, 0, 0, G)
            model.add_hop(-t2, 1, 1, G)

        self.model = model

    def test_no_uc_unity(self):
        self.createH(0.1, 0.2)
        new_model = self.model.change_uc(uc=np.identity(3))
        self.assertFullAlmostEqual(new_model.uc, self.model.uc)
        self.assertFullAlmostEqual(self.model.pos, new_model.pos)
            
    def test_uc_unity(self):
        self.createH(0.1, 0.2, uc=np.diag([1, 2, 3]))
        new_model = self.model.change_uc(uc=np.identity(3))
        self.assertFullAlmostEqual(new_model.uc, self.model.uc)
        self.assertFullAlmostEqual(self.model.pos, new_model.pos)
            
    def test_uc_1(self):
        self.createH(0.1, 0.2, uc=np.diag([1, 2, 3]))
        new_model = self.model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

        res_uc = in_place_replace(new_model.uc)
        res_pos = in_place_replace(new_model.pos)
        self.assertFullAlmostEqual(new_model.uc, res_uc)
        self.assertFullAlmostEqual(new_model.pos, res_pos)
            
    def test_uc_2(self):
        self.createH(0.1, 0.2, uc=np.array([[0, 1, 0], [2, 0, 1], [1, 0, 1]]))
        new_model = self.model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

        res_uc = in_place_replace(new_model.uc)
        res_pos = in_place_replace(new_model.pos)
        self.assertFullAlmostEqual(new_model.uc, res_uc)
        self.assertFullAlmostEqual(new_model.pos, res_pos)
        
    def test_uc_hamilton_unity(self):
        self.createH(0.1, 0.2, uc=np.diag([1, 2, 3]))
        new_model = self.model.change_uc(uc=np.identity(3))

        res = in_place_replace(new_model.hamilton([0.1, 0.4, 0.7]))
        self.assertFullAlmostEqual(res, new_model.hamilton([0.1, 0.4, 0.7]))
        
    def test_uc_hamilton_1(self):
        self.createH(0.1, 0.2, uc=np.diag([1, 2, 3]))
        new_model = self.model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

        res = in_place_replace(new_model.hamilton([0.1, 0.4, 0.7]))
        self.assertFullAlmostEqual(res, new_model.hamilton([0.1, 0.4, 0.7]))
        
    def test_uc_hamilton_2(self):
        self.createH(0.1, 0.2, uc=np.array([[0, 1, 0], [2, 0, 1], [1, 0, 1]]))
        new_model = self.model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

        res = in_place_replace(new_model.hamilton([0.1, 0.4, 0.7]))
        self.assertFullAlmostEqual(res, new_model.hamilton([0.1, 0.4, 0.7]))
            
    def test_error(self):
        self.createH(0.1, 0.2, uc=np.identity(3))
        self.assertRaises(ValueError, self.model.change_uc, uc=np.diag([2, 1, 1]))
    

if __name__ == "__main__":
    unittest.main()
