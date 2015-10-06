#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    uc_mapping.py

from common import *

import numpy as np

class UcMappingTestCase(CommonTestCase):
    def uc_mapping(self, t1, t2, k):
        builder = tbmodels.Builder()

        # create the two atoms
        builder.add_atom([1], [0, -0.1, 0.], 1)
        builder.add_atom([-1], [2.15, 0.5, -0.2], 0)
        builder.add_atom([0.], [1.75, 0.25, 0.6], 0)
        builder.add_atom([0.], [0.75, -0.15, 0.6], 0)

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
        model = builder.create()
        return model.hamilton(k)

    def test_uc_mapping(self):
        vars = [
            [0.1, 0.2, [0., 0.2, 0.7]],
            [0.9, -0.2, [0.9, 0.2, 0.7]],
            [0.1, 0.3, [0.1, 0.12, -0.7]],
        ]
        res0 = in_place_replace(self.uc_mapping(*vars[0]))
        res1 = in_place_replace(self.uc_mapping(*vars[1]))
        res2 = in_place_replace(self.uc_mapping(*vars[2]))
        self.assertFullAlmostEqual(res0, self.uc_mapping(*vars[0]))
        self.assertFullAlmostEqual(res1, self.uc_mapping(*vars[1]))
        self.assertFullAlmostEqual(res2, self.uc_mapping(*vars[2]))

if __name__ == "__main__":
    unittest.main()
