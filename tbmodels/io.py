#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import numbers
from collections.abc import Iterable
from functools import singledispatch

import h5py
import numpy as np
from fsc.export import export

from ._tb_model import Model

@export
def save(obj, file_path):
    """
    Saves TBmodels objects (or nested lists thereof) to a file, using the HDF5 format.
    """
    with h5py.File(file_path, 'w') as hf:
        _encode(obj, hf)

@singledispatch
def _encode(obj, hf):
    raise ValueError('Cannot encode object of type {}'.format(type(obj)))

@_encode.register(str)
@_encode.register(numbers.Complex)
@_encode.register(np.ndarray)
def _(obj, hf):
    hf['val'] = obj

@_encode.register(Iterable)
def _(obj, hf):
    for i, part in enumerate(obj):
        sub_group = hf.create_group(str(i))
        _encode(part, sub_group)

@_encode.register(Model)
def _(obj, hf):
    obj.to_hdf5(hf)

@export
def load(file_path):
    with h5py.File(file_path, 'r') as hf:
        return _decode(hf)

def _decode(hf):
    if 'tb_model' in hf:
        return _decode_model(hf)
    elif 'val' in hf:
        return _decode_val(hf)
    elif '0' in hf:
        return _decode_iterable(hf)
    else:
        raise ValueError('File structure not understood.')

def _decode_iterable(hf):
    return [_decode(hf[key]) for key in sorted(hf, key=int)]

def _decode_model(hf):
    return Model.from_hdf5(hf)

def _decode_val(hf):
    return hf['val'].value
