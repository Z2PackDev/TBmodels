#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Helper module wrapping scipy.sparse matrix types to make them
convertible directly to numpy arrays by implementing the __array__
method.
"""

import scipy.sparse as sp


class ArrayConvertible:
    """
    Base class for sparse matrix types that should be directly convertible
    to numpy arrays.
    """

    def __array__(self):
        return self.toarray()  # pylint: disable=no-member

    # Because transpose / conjugate return scipy sparse arrays.
    def transpose(self):
        return self.__class__(super().transpose())  # pylint: disable=no-member

    def conjugate(self):
        return self.__class__(super().conjugate())  # pylint: disable=no-member


class csr(ArrayConvertible, sp.csr_matrix):  # pylint: disable=invalid-name
    """
    Wrapper for CSR matrices to be array-convertible.
    """


class coo(ArrayConvertible, sp.coo_matrix):  # pylint: disable=invalid-name
    """
    Wrapper for COO matrices to be array-convertible.
    """


class lil(ArrayConvertible, sp.lil_matrix):  # pylint: disable=invalid-name
    """
    Wrapper for LIL matrices to be array-convertible.
    """
