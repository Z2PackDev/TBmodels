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

        res = in_place_replace(str(self.model))
        self.assertEqual(str(self.model), res)

if __name__ == "__main__":
    unittest.main()
