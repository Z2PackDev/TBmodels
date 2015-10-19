#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    str.py

from common import *
from ptools.sparse_matrix import csr

import numpy as np

class StrTestCase(SimpleModelTestCase):
    def test_print(self):
        self.createH(0.1, 0.2)
        self.model.pos = np.array([[0., 0., 0.5], [-0.2, -0.1, 0.]])
        self.model.change_uc(np.array([[1, 2, 0], [0, 10, 0], [0, 0, 0.1]]))

        res = '[general]\ndim = 3\nocc = 1\nsize = 2\n\n[pos]\n 0                                                             0                                                             0.5\n-0.200000000000000011102230246251565404236316680908203125     -0.1000000000000000055511151231257827021181583404541015625     0\n\n\n[hop]\n(0 0 0)\n     0     0     0.5+0j\n     0     1     0.1000000000000000055511151231257827021181583404541015625+0j\n     1     1    -0.5+0j\n\n(0 1 0)\n     0     0     0.200000000000000011102230246251565404236316680908203125+0j\n     1     0     0+0.1000000000000000055511151231257827021181583404541015625j\n     1     1    -0.200000000000000011102230246251565404236316680908203125+0j\n\n(1 0 0)\n     0     0     0.200000000000000011102230246251565404236316680908203125+0j\n     1     0     0-0.1000000000000000055511151231257827021181583404541015625j\n     1     1    -0.200000000000000011102230246251565404236316680908203125+0j\n\n(1 1 0)\n     1     0    -0.1000000000000000055511151231257827021181583404541015625+0j\n'
        self.assertEqual(str(self.model), res)

    def test_load(self):
        self.createH(0.1, 0.3)
        new_model = tbmodels.read_model.tbm(str(self.model))
        self.assertFullEqual(self.model.uc, new_model.uc)
        self.assertFullEqual(self.model.pos, new_model.pos)
        self.assertFullEqual(
            {key: np.array(val) for key, val in self.model.hop.items()},
            {key: np.array(val) for key, val in new_model.hop.items()}
        )
        #~ self.assertFullEqual(self.model.hop, new_model.hop)

if __name__ == "__main__":
    unittest.main()
