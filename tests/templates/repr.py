#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    repr.py

from common import *
from ptools.sparse_matrix import csr

import numpy as np

class ReprTestCase(SimpleModelTestCase):
    def test_print(self):
        self.createH(0.1, 0.2)
        
        new_model = eval(repr(self.model))

        self.assertFullAlmostEqual(self.model.hamilton([0.1, 0.6, 0.9]), new_model.hamilton([0.1, 0.6, 0.9]))

if __name__ == "__main__":
    unittest.main()
