#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    06.10.2015 17:35:33 CEST
# File:    sparse_matrix.py

import numpy as np
import scipy.sparse as sp


class ArrayConvertible(object):
    def __array__(self):
        return self.toarray()

    # Because transpose / conjugate return scipy sparse arrays.
    def transpose(self):
        return self.__class__(super(ArrayConvertible, self).transpose())

    def conjugate(self):
        return self.__class__(super(ArrayConvertible, self).conjugate())


class csr(ArrayConvertible, sp.csr_matrix):
    def __repr__(self):
        res = (
            'csr((' + '[' + ', '.join(str(x)
                                      for x in self.data) + '], ' + '[' +
            ', '.join(str(x) for x in self.indices) + '], ' + '[' + ', '.join(
                str(x) for x in self.indptr
            ) + ']), ' + 'shape={0.shape}, dtype=np.{0.dtype})'.format(self)
        )
        return res

    # This is here because scipy throws NotImplementedError
    # Is not needed for newer versions of scipy
    def __iadd__(self, other):
        return self + other


class coo(ArrayConvertible, sp.coo_matrix):
    pass


class lil(ArrayConvertible, sp.lil_matrix):
    pass
