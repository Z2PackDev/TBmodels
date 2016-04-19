#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    18.02.2016 18:07:11 MST
# File:    conftest.py

import json
import pytest
import numbers
from collections.abc import Iterable
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import numpy as np

import tbmodels

@pytest.fixture
def test_name(request):
    """Returns module_name.function_name for a given test"""
    return request.module.__name__ + '/' + request._parent_request._pyfuncitem.name

@singledispatch
def encode(obj):
    raise TypeError('cannot JSONify {} object {}'.format(type(obj), obj))

@encode.register(bool)
@encode.register(np.bool_)
def _(obj):
    return bool(obj)

@encode.register(numbers.Integral)
def _(obj):
    return int(obj)

@encode.register(numbers.Real)
def _(obj):
    return float(obj)
    
@encode.register(numbers.Complex)
def _(obj):
    return dict(__complex__=True, real=encode(obj.real), imag=encode(obj.imag))

@encode.register(str)
def _(obj):
    return obj

@encode.register(Iterable)
def _(obj):
    return list(obj)
        
@pytest.fixture
def compare_data(request, test_name, scope="session"):
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""
    def inner(compare_fct, data, tag=None):
        full_name = test_name + (tag or '')
        val = request.config.cache.get(full_name, None)
        if val is None:
            request.config.cache.set(full_name, json.loads(json.dumps(data, default=encode)))
            raise ValueError('Reference data does not exist.')
        else:
            assert compare_fct(val, json.loads(json.dumps(data, default=encode))) # get rid of json-specific quirks
    return inner

@pytest.fixture
def compare_equal(compare_data):
    return lambda data, tag=None: compare_data(lambda x, y: x == y, data, tag)
