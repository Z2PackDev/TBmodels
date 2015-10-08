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

class TrsTestCase(SimpleModelTestCase):
    def createH(self, *args, **kwargs):
        super(TrsTestCase, self).createH(*args, **kwargs)
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
