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

class TrsTestCase(BuildDirTestCase):

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

    def test_trs_inplace(self):
        self.createH(0.2, 0.3)
        # create a second TRS model by in-place replacing
        model2 = copy.deepcopy(self.model)
        model2.trs(in_place=True)
        self.assertFullAlmostEqual(self.trs_model.hamilton([0.1, 0.2, 0.7]), model2.hamilton([0.1, 0.2, 0.7]))

if __name__ == "__main__":
    unittest.main()
