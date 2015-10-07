#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    hr_consistency.py

from common import *

import numpy as np

class HrConsistencyTestCase(CommonTestCase):
    def test(self):
        model = tbmodels.HrModel('./samples/hr_hamilton.dat', occ=28)
        lines_new = model.to_hr().split('\n')
        with open('./samples/hr_hamilton.dat', 'r') as f:
            lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
        self.assertFullEqual(lines_new[1:], lines_old[1:])

if __name__ == "__main__":
    unittest.main()
