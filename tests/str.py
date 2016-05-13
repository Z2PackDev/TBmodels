#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    str.py

from common import *
from tbmodels._ptools.sparse_matrix import csr

import numpy as np

class StrTestCase(SimpleModelTestCase):
    def test_print(self):
        self.maxDiff = None # to print the full diff when test fails
        self.createH(0.1, 0.2)
        self.model.pos = np.array([[0., 0., 0.5], [-0.2, -0.1, 0.]])
        self.model.change_uc(np.array([[1, 2, 0], [0, 10, 0], [0, 0, 0.1]]))

        res = in_place_replace(str(self.model))
        self.assertEqual(str(self.model), res)

    def test_load(self):
        self.createH(0.1, 0.3)
        new_model = tbmodels.read_model.tbm(str(self.model))
        self.assertFullEqual(self.model.dim, new_model.dim)
        self.assertFullEqual(self.model.size, new_model.size)
        self.assertFullEqual(self.model.uc, new_model.uc)
        self.assertFullEqual(self.model.pos, new_model.pos)
        self.assertFullEqual(
            {key: np.array(val) for key, val in self.model.hop.items()},
            {key: np.array(val) for key, val in new_model.hop.items()}
        )

if __name__ == "__main__":
    unittest.main()
