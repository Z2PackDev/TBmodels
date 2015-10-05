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
        res = array([[ 1.67082039+0.j       ,  0.18914915+0.2902113j],
       [ 0.18914915-0.2902113j, -1.67082039+0.j       ]])

        self.assertFullAlmostEqual(res, self.model.hamilton([0.1, 0.2, 0.7]))

    def test_trs(self):
        self.createH(0.2, 0.3)
        res = array([[ 0.70000000+0.j        ,  0.44596495-0.03339549j,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.44596495+0.03339549j, -0.70000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.70000000+0.j        , -0.16957175+0.4138181j ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
        -0.16957175-0.4138181j , -0.70000000+0.j        ]])

        self.assertFullAlmostEqual(res, self.trs_model.hamilton([0.4, -0.2, 0.1]))


if __name__ == "__main__":
    unittest.main()
