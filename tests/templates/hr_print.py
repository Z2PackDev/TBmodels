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

        builder = tbmodels.Builder()

        # create the two atoms
        builder.add_atom([1], [0, 0, 0.], 1)
        builder.add_atom([-1], [0.5, 0.5, 0.2], 0)
        builder.add_atom([0.], [0.75, 0.15, 0.6], 0)

        # add hopping between different atoms
        builder.add_hopping(((0, 0), (1, 0)),
                           tbmodels.helpers.combine([0, -1], [0, -1], 0),
                           t1,
                           phase=[1, -1j, 1j, -1])

        # add hopping between neighbouring orbitals of the same type
        builder.add_hopping(((0, 0), (0, 0)),
                           tbmodels.helpers.neighbours([0, 1],
                                                        forward_only=True),
                           t2,
                           phase=[1])
        builder.add_hopping(((1, 0), (1, 0)),
                           tbmodels.helpers.neighbours([0, 1],
                                                        forward_only=True),
                           -t2,
                           phase=[1])
        self.model = builder.create(uc=uc)

    def test0(self):
        self.createH(0.1, 0.211)
        res = in_place_replace(self.model.to_hr())
        self.assertEqual(res, self.model.to_hr())
    

if __name__ == "__main__":
    unittest.main()
