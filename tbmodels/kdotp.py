# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the :class:`.KdotpModel` class for k.p models.
"""

import typing as ty

import numpy as np
import scipy.linalg as la

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('tbmodels.kdotp_model', check_on_load=False)
class KdotpModel(SimpleHDF5Mapping):
    """
    A class describing a k.p model.

    Parameters
    ----------
    taylor_coefficients:
        A mapping containing the taylor coefficients of the k.p model.
        The keys are tuples which describe the power of the k-vector
        components, and the values are the corresponding matrices.

        Example:
            (1, 0, 2): [[1, 0], [0, -1]]
            describes k_x * k_z**2 * sigma_z
    """
    HDF5_ATTRIBUTES = ['taylor_coefficients']

    def __init__(
        self, taylor_coefficients: ty.Mapping[ty.Collection[int], ty.Collection[ty.Collection[complex]]]
    ) -> None:
        for mat in taylor_coefficients.values():
            if not np.allclose(mat, np.array(mat).T.conj()):
                raise ValueError('The provided Taylor coefficient {} is not hermitian'.format(mat))
        self.taylor_coefficients = {
            tuple(key): np.array(mat, dtype=complex)
            for key, mat in taylor_coefficients.items()
        }

    def hamilton(self, k: ty.Collection[float]) -> np.ndarray:
        return sum(np.prod(np.array(k)**np.array(k_powers)) * mat for k_powers, mat in self.taylor_coefficients.items())

    def eigenval(self, k: ty.Collection[float]) -> np.ndarray:
        return la.eigvalsh(self.hamilton(k))
