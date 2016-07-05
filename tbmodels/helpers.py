#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    04.06.2015 14:26:21 CEST
# File:    helpers.py

"""
TODO
"""

import numbers
import contextlib
from functools import singledispatch
from collections.abc import Iterable

import numpy as np

from ._ptools import sparse_matrix as sp

from ._tb_model import Model

def matrix_to_hop(mat, orbitals=None, R=(0, 0, 0), multiplier=1.):
    r"""
    Turns a square matrix into a series of hopping terms.

    :param mat: The matrix to be converted.

    :param orbitals:    Indices of the orbitals that make up the basis w.r.t. which the matrix is defined. By default (``orbitals=None``), the first ``len(mat)`` orbitals are used.
    :type orbitals:     list

    :param R:   Lattice vector for all the hopping terms.
    :type R:    list

    :param multiplier:  Multiplicative constant for the hoppings strength.
    :type multiplier: float / complex
    """
    if orbitals is None:
        orbitals = list(range(len(mat)))
    hop = []
    for i, row in enumerate(mat):
        for j, x in enumerate(row):
            hop.append([multiplier * x, orbitals[i], orbitals[j], np.array(R, dtype=int)])
    return hop

#-------------------------------ENCODING--------------------------------#

@singledispatch
def encode(obj):
    """
    Encodes TBmodels types into JSON / msgpack - compatible types.
    """
    raise TypeError('cannot JSONify {} object {}'.format(type(obj), obj))

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

@encode.register(Iterable)
def _(obj):
    return list(obj)

@encode.register(Model)
def _(obj):
    return dict(
        __tb_model__=True,
        uc=obj.uc,
        occ=obj.occ,
        size=obj.size,
        dim=obj.dim,
        pos=obj.pos,
        hop=_encode_hoppings(obj.hop)
    )

def _encode_hoppings(hoppings):
    return dict(
        __hoppings__=[
            (R, (mat.data, mat.indices, mat.indptr), mat.shape)
            for R, mat in hoppings.items()
        ]
    )

#-------------------------------DECODING--------------------------------#

def decode_tb_model(obj):
    del obj['__tb_model__']
    return Model(contains_cc=False, **obj)
    
def decode_hoppings(obj):
    return {
        tuple(R): sp.csr(tuple(mat), shape=shape)
        for R, mat, shape in obj['__hoppings__']
    }
    
def decode_complex(obj):
    return complex(obj['real'], obj['imag'])

def decode(dct):
    with contextlib.suppress(AttributeError):
        dct = {k.decode('utf-8'): v for k, v in dct.items()}
    special_markers = [key for key in dct.keys() if key.startswith('__')]
    if len(special_markers) == 1:
        name = special_markers[0].strip('__')
        return globals()['decode_' + name](dct)
    else:
        return dct
