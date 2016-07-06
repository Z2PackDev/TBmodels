#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    18.02.2016 18:07:11 MST
# File:    conftest.py

import json
import pytest
import numbers
import contextlib
from collections.abc import Iterable
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import numpy as np

import tbmodels

#--------------------------ENCODING-------------------------------------#

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
        
#--------------------------DECODING-------------------------------------#

@singledispatch
def decode(obj):
    return obj

@decode.register(str)
def _(obj):
    return obj

@decode.register(Iterable)
def _(obj):
    return [decode(x) for x in obj]

def decode_complex(obj):
    return complex(obj['real'], obj['imag'])


@decode.register(dict)
def _(obj):
    with contextlib.suppress(AttributeError):
        obj = {k.decode('utf-8'): v for k, v in obj.items()}
    special_markers = [key for key in obj.keys() if key.startswith('__')]
    if len(special_markers) == 1:
        name = special_markers[0].strip('__')
        return globals()['decode_' + name](obj)
    else:
        return obj

#--------------------------FIXTURES-------------------------------------#

@pytest.fixture
def test_name(request):
    """Returns module_name.function_name for a given test"""
    return request.module.__name__ + '/' + request._parent_request._pyfuncitem.name

@pytest.fixture
def compare_data(request, test_name, scope="session"):
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""
    def inner(compare_fct, data, tag=None):
        full_name = test_name + (tag or '')
        
        # get rid of json-specific quirks
        # store as string because I cannot add the decoder to the pytest cache
        data_str = json.dumps(data, default=encode)
        data = json.loads(data_str, object_hook=decode)
        val = json.loads(request.config.cache.get(full_name, 'null'), object_hook=decode)

        if val is None:
            request.config.cache.set(full_name, data_str)
            raise ValueError('Reference data does not exist.')
        else:
            assert compare_fct(val, data) 
    return inner

@pytest.fixture
def compare_equal(compare_data):
    return lambda data, tag=None: compare_data(lambda x, y: x == y, data, tag)
    
@pytest.fixture
def models_equal():
    def inner(model1, model2):
        assert model1.size == model2.size
        assert model1.dim == model2.dim
        assert np.array(model1.uc == model2.uc).all()
        assert model1.occ == model2.occ
        for k in model1.hop.keys() | model2.hop.keys():
            print('k:', k)
            print('model1:\n', np.array(model1.hop[k]))
            print('model2:\n', np.array(model2.hop[k]))
            assert (np.array(model1.hop[k]) == np.array(model2.hop[k])).all()
        assert (model1.pos == model2.pos).all()
    return inner
    
@pytest.fixture
def models_close():
    def inner(model1, model2):
        assert model1.size == model2.size
        assert model1.dim == model2.dim
        if model1.uc is None:
            assert model1.uc == model2.uc
        else:
            assert np.isclose(model1.uc, model2.uc).all()
        assert model1.occ == model2.occ
        for k in model1.hop.keys() | model2.hop.keys():
            print('k:', k)
            print('model1:\n', np.array(model1.hop[k]))
            print('model2:\n', np.array(model2.hop[k]))
            assert np.isclose(np.array(model1.hop[k]), np.array(model2.hop[k])).all()
        if model1.pos is None:
            assert model1.pos == model2.pos
        else:
            assert np.isclose(model1.pos, model2.pos).all()
    return inner

