# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Module defining k.p models. Note that this module should be considered temporary,
and should not be relied on having a fixed import position. Since it does not
treat tight-binding models, it might be moved to a separate package.
"""

import numpy as np
import scipy.linalg as la

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('tbmodels.kdotp_model', check_on_load=False)
class KdotpModel(SimpleHDF5Mapping):
    """
    A class describing a k.p model.

    .. note:: This feature is experimental, and may be moved to a separate
        package in the future.

    :param taylor_coefficients: A mapping containing the taylor coefficients of
        the k.p model. The keys are tuples which describe the power of the
        k-vector components, and the values are the corresponding matrices.
        Example:
            (1, 0, 2): [[1, 0], [0, -1]]
            describes k_x * k_z**2 * sigma_z
    :type taylor_coefficients: dict
    """
    HDF5_ATTRIBUTES = ['taylor_coefficients']

    def __init__(self, taylor_coefficients):
        for mat in taylor_coefficients.values():
            if not np.allclose(mat, np.array(mat).T.conj()):
                raise ValueError('The provided Taylor coefficient {} is not hermitian'.format(mat))
        self.taylor_coefficients = {
            tuple(key): np.array(mat, dtype=complex)
            for key, mat in taylor_coefficients.items()
        }

    def hamilton(self, k):
        return sum(np.prod(np.array(k)**np.array(pow)) * mat for pow, mat in self.taylor_coefficients.items())

    def eigenval(self, k):
        return la.eigvalsh(self.hamilton(k))
