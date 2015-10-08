#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    arithmetics.py

from common import *

import os
import copy
import types
import shutil

class ArithmeticTestCase(SimpleModelTestCase):
    #~ def createH(self, t1, t2):
#~ 
        #~ builder = tbmodels.Builder()
#~ 
        #~ # create the two atoms
        #~ builder.add_atom([1], [0, 0, 0], 1)
        #~ builder.add_atom([-1], [0.5, 0.5, 0], 0)
#~ 
        #~ # add hopping between different atoms
        #~ builder.add_hopping(((0, 0), (1, 0)),
                           #~ tbmodels.helpers.combine([0, -1], [0, -1], 0),
                           #~ t1,
                           #~ phase=[1, -1j, 1j, -1])
#~ 
        #~ # add hopping between neighbouring orbitals of the same type
        #~ builder.add_hopping(((0, 0), (0, 0)),
                           #~ tbmodels.helpers.neighbours([0, 1],
                                                        #~ forward_only=True),
                           #~ t2,
                           #~ phase=[1])
        #~ builder.add_hopping(((1, 0), (1, 0)),
                           #~ tbmodels.helpers.neighbours([0, 1],
                                                        #~ forward_only=True),
                           #~ -t2,
                           #~ phase=[1])
        #~ return builder.create()

    def test_add_1(self):
        model1 = self.createH(0.2, 0.3)
        model2 = self.createH(0.0, 0.1)
        model3 = model1 + model2
        res = array([[ 2.89442719+0.j       ,  0.18914915+0.2902113j],
       [ 0.18914915-0.2902113j, -2.89442719+0.j       ]])

        self.assertFullAlmostEqual(res, model3.hamilton([0.1, 0.2, 0.7]))

    def test_add_2(self):
        model1 = self.createH(0.2, 0.3)
        model2 = self.createH(0.7, -0.1)
        model3 = model1 + model2
        res = array([[ 2.44721360+0.j        ,  0.85117116+1.30595086j],
       [ 0.85117116-1.30595086j, -2.44721360+0.j        ]])

        self.assertFullAlmostEqual(res, model3.hamilton([0.1, 0.2, 0.7]))

    def test_sub_1(self):
        model1 = self.createH(0.2, 0.3)
        model2 = self.createH(0.0, 0.1)
        model3 = model1 - model2
        res = array([[ 0.44721360+0.j       ,  0.18914915+0.2902113j],
       [ 0.18914915-0.2902113j, -0.44721360+0.j       ]])

        self.assertFullAlmostEqual(res, model3.hamilton([0.1, 0.2, 0.7]))

    def test_sub_2(self):
        model1 = self.createH(0.2, 0.3)
        model2 = self.createH(0.7, -0.1)
        model3 = -model1 - model2
        res = array([[-2.44721360+0.j        , -0.85117116-1.30595086j],
       [-0.85117116+1.30595086j,  2.44721360+0.j        ]])

        self.assertFullAlmostEqual(res, model3.hamilton([0.1, 0.2, 0.7]))

    def test_mul_1(self):
        model1 = self.createH(0.2, 0.3)
        model2 = model1 * 0.2
        res = array([[ 0.33416408+0.j        ,  0.03782983+0.05804226j],
       [ 0.03782983-0.05804226j, -0.33416408+0.j        ]])

        self.assertFullAlmostEqual(res, model2.hamilton([0.1, 0.2, 0.7]))

    def test_mul_2(self):
        model1 = self.createH(0.7, -0.1)
        model2 = 0.2 * model1
        res = array([[ 0.15527864+0.j        ,  0.13240440+0.20314791j],
       [ 0.13240440-0.20314791j, -0.15527864+0.j        ]])

        self.assertFullAlmostEqual(res, model2.hamilton([0.1, 0.2, 0.7]))

    def test_div(self):
        model1 = self.createH(0.7, -0.1)
        model2 = model1 / 3
        model3 = model1 * (1. / 3)
        res = array([[ 0.25879773+0.j        ,  0.22067400+0.33857985j],
       [ 0.22067400-0.33857985j, -0.25879773+0.j        ]])

        self.assertFullAlmostEqual(res, model2.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(model2.hamilton([0.1, 0.21783, 0.2]), model3.hamilton([0.1, 0.21783, 0.2]))

if __name__ == "__main__":
    unittest.main()
