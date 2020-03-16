# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines functions for saving and loading in HDF5 format.
"""

import copy
import warnings

import h5py
import fsc.hdf5_io
from fsc.export import export

from . import _legacy_decode

__all__ = ['save']

save = copy.deepcopy(fsc.hdf5_io.save)  # pylint: disable=invalid-name
save.__doc__ = "Save TBmodels objects to a HDF5 file. Compatible with all types registered through :py:mod:`fsc.hdf5_io`."


@export
def load(file_path):
    """
    Load TBmodels objects from an HDF5 file. Compatible with all types registered through :py:mod:`fsc.hdf5_io`.
    """
    try:
        return fsc.hdf5_io.load(file_path)
    except ValueError:
        warnings.warn(
            f"The loaded file '{file_path}' is stored in an outdated "
            "format. Consider loading and storing the file to update it.", DeprecationWarning
        )
        with h5py.File(file_path, 'r') as hdf5_handle:
            return _legacy_decode._decode(hdf5_handle)  # pylint: disable=protected-access
