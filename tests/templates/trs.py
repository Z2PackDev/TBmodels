#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    trs.py

from common import *

import os
import copy
import types
import shutil

class TrsTestCase(CommonTestCase):

    def createH(self, t1, t2):
        model = tbmodels.Model(size=2, pos=[[0, 0, 0], [0.5, 0.5, 0]], occ=1)

        model.add_on_site(1., 0)
        model.add_on_site(-1., 1)

        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hop(t1 * phase, 0, 1, G)

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hop(t2, 0, 0, G)
            model.add_hop(-t2, 1, 1, G)
            
        self.model = model
        self.trs_model = self.model.trs()
        
    # this test may produce false negatives due to small numerical differences
    def test_notrs(self):
        self.createH(0.2, 0.3)
        res = in_place_replace(self.model.hamilton([0.1, 0.2, 0.7]))

        self.assertFullAlmostEqual(res, self.model.hamilton([0.1, 0.2, 0.7]))

    def test_trs(self):
        self.createH(0.2, 0.3)
        res = in_place_replace(self.trs_model.hamilton([0.4, -0.2, 0.1]))

        self.assertFullAlmostEqual(res, self.trs_model.hamilton([0.4, -0.2, 0.1]))

if __name__ == "__main__":
    unittest.main()
