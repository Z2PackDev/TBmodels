#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    strip.py

from common import *

import os
import copy
import types
import shutil

class StripTestCase(BuildDirTestCase):

    def createH(self, t1, t2):

        builder = tbmodels.Builder()

        # create the two atoms
        builder.add_atom([1], [0, 0, 0], 1)
        builder.add_atom([-1], [0.5, 0.5, 0], 0)

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
        self.model = builder.create()

    def test_trs_inplace(self):
        self.createH(0.2, 0.3)
        k = [0.1, -0.6, 0.2]
        res = self.model.hamilton(k)
        self.model.strip()
        self.assertFullAlmostEqual(res, self.model.hamilton(k))

if __name__ == "__main__":
    unittest.main()
