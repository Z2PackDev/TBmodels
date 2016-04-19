#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    uc_mapping.py

from common import *
import collections as co

import numpy as np

class UcMappingTestCase(CommonTestCase):
    def uc_mapping(self, t1, t2, k):
        hoppings = co.defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            mat = np.zeros((4, 4), dtype=complex)
            mat[0, 1] += phase * t1
            hoppings[tuple(G)] += mat
            hoppings[tuple([-x for x in G])] += mat.conjugate().transpose()

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            mat = np.zeros((4, 4), dtype=complex)
            mat[0, 0] = t2
            mat[1, 1] = -t2
            hoppings[tuple(G)] += mat
            hoppings[tuple(-x for x in G)] += mat.conjugate().transpose()

        model = tbmodels.Model(
            on_site=[1, -1, 0, 0],
            hop=hoppings,
            pos=[
                [0, -0.1, 0.],
                [2.15, 0.5, -0.2],
                [1.75, 0.25, 0.6],
                [0.75, -0.15, 0.6],
            ],
            occ=1,
            )

        return model.hamilton(k)

    def test_uc_mapping(self):
        vars = [
            [0.1, 0.2, [0., 0.2, 0.7]],
            [0.9, -0.2, [0.9, 0.2, 0.7]],
            [0.1, 0.3, [0.1, 0.12, -0.7]],
        ]
        res0 = array([[ 1.52360680+0.j        ,  0.02600735-0.16420395j,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.02600735+0.16420395j, -1.52360680+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ]])
        res1 = array([[ 0.55278640+0.j        , -0.55623059-1.45623059j,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [-0.55623059+1.45623059j, -0.55278640+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ]])
        res2 = array([[ 1.92279137+0.j        ,  0.05770661+0.11436794j,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.05770661-0.11436794j, -1.92279137+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ],
       [ 0.00000000+0.j        ,  0.00000000+0.j        ,
         0.00000000+0.j        ,  0.00000000+0.j        ]])
        self.assertFullAlmostEqual(res0, self.uc_mapping(*vars[0]))
        self.assertFullAlmostEqual(res1, self.uc_mapping(*vars[1]))
        self.assertFullAlmostEqual(res2, self.uc_mapping(*vars[2]))

if __name__ == "__main__":
    unittest.main()
