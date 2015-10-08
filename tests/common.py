#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:18:11 CEST
# File:    common.py

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/../')
import tbmodels
# for create_tests
try:
    from tbmodels.ptools.replace import *
except ImportError:
    pass

import types
import shutil
import inspect
import warnings
warnings.simplefilter('always')
import traceback

if sys.version <= '2.6.x':
    import unittest2 as unittest
else:
    import unittest

# ugly hack to enable in-place replacement of arrays
from numpy import array

def assertFullAlmostEqual(TestCase, x, y):
    """
    Compares for almost equality
    """
    # str
    if isinstance(x, str):
        TestCase.assertEqual(x, y)
    # dict
    elif hasattr(x, 'keys'):
        if not(sorted(x.keys()) == sorted(y.keys())):
            TestCase.fail(msg="dicts don't have the same keys")
        for key in x.keys():
            TestCase.assertFullAlmostEqual(x[key], y[key])
    # list, tuple
    elif hasattr(x, '__iter__'):
        if len(x) != len(y):
            TestCase.fail(msg='length of objects is not equal')
        for xval, yval in zip(x, y):
            TestCase.assertFullAlmostEqual(xval, yval)
    # rest
    else:
        try:
            TestCase.assertAlmostEqual(x, y)
        except TypeError:
            TestCase.assertEqual(x, y)

def assertFullEqual(TestCase, x, y):
    """
    Compares for almost equality
    """
    # str
    if isinstance(x, str):
        TestCase.assertEqual(x, y)
    # dict
    elif hasattr(x, 'keys'):
        if not(sorted(x.keys()) == sorted(y.keys())):
            TestCase.fail(msg="dicts don't have the same keys")
        for key in x.keys():
            TestCase.assertFullEqual(x[key], y[key])
    # list, tuple
    elif hasattr(x, '__iter__'):
        if len(x) != len(y):
            TestCase.fail(msg='length of objects is not equal')
        for xval, yval in zip(x, y):
            TestCase.assertFullEqual(xval, yval)
    # rest
    else:
        TestCase.assertEqual(x, y)

def assertWccConv(TestCase, x, y, epsilon = 1e-6):
    """
    Compares two WCC lists
    """
    assert(len(x) == len(y))
    for x_wcc, y_wcc in zip(x, y):
        TestCase.assertTrue(tbmodels._core._convcheck(x_wcc, y_wcc, epsilon))

class CommonTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CommonTestCase, self).__init__(*args, **kwargs)
        self.assertFullAlmostEqual = types.MethodType(
            assertFullAlmostEqual, self)
        self.assertFullEqual = types.MethodType(
            assertFullEqual, self)
        self.assertWccConv = types.MethodType(
            assertWccConv, self)

class SimpleModelTestCase(CommonTestCase):
    def createH(self, t1, t2, uc=None):
        model = tbmodels.Model(size=2, on_site=[1, -1], pos=[[0, 0, 0], [0.5, 0.5, 0]], occ=1, uc=uc)

        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hop(t1 * phase, 0, 1, G)

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hop(t2, 0, 0, G)
            model.add_hop(-t2, 1, 1, G)
            
        self.model = model
        return self.model
    
#~ class BuildDirTestCase(CommonTestCase):
    #~ def __init__(self, *args, **kwargs):
        #~ self._name = traceback.extract_stack()[0][0].split('.')[0]
        #~ if self._name in ['test', 'create_tests']:
            #~ self._name = re.search("'([\w]+).[\w]+'", str(type(self))).group(1)
        #~ self._build_folder = 'build/' + self._name
        #~ try:
            #~ shutil.rmtree(self._build_folder)
        #~ except OSError:
            #~ pass
        #~ os.mkdir(self._build_folder)
        #~ super(BuildDirTestCase, self).__init__(*args, **kwargs)
