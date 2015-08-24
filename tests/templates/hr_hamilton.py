#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

from common import *

import numpy as np

class HrHamiltonTestCase(CommonTestCase):
    
    def testH(self):
        model = tbmodels.HrModel('./samples/hr_hamilton.dat', occ=28)
        k = [0.412351236, 0.872372, 0.31235123]
        H = in_place_replace(list(model.hamilton(k)))
        self.assertFullAlmostEqual(model.hamilton(k), H)
        
    def test_error(self):
        self.assertRaises(ValueError, tbmodels.HrModel, './samples/hr_hamilton.dat', occ=28, pos=[[1., 1., 1.]])

if __name__ == "__main__":
    unittest.main()
