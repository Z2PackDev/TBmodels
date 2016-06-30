#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    change_uc.py

from common import *

import numpy as np

class HrPrintTestCase(CommonTestCase):
    def createH(self, t1, t2, uc=None):

        model = tbmodels.Model(on_site=[1, -1, 0], pos=[[0, 0, 0], [0.5, 0.5, 0], [0.75, 0.15, 0.6]], occ=1, uc=uc, dim=3)

        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hopping(t1 * phase, 0, 1, G)

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hopping(t2, 0, 0, G)
            model.add_hopping(-t2, 1, 1, G)
            
        self.model = model
        return self.model

    def test0(self):
        self.createH(0.1, 0.211)
        res = in_place_replace(self.model.to_hr())
        self.assertFullEqual(res.split('\n')[1:], self.model.to_hr().split('\n')[1:])
    

if __name__ == "__main__":
    unittest.main()
