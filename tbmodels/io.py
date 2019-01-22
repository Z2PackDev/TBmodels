# © 2015-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines functions for saving and loading in HDF5 format.
"""

import h5py
import fsc.hdf5_io
from fsc.export import export

from . import _legacy_decode

__all__ = ['save']

save = fsc.hdf5_io.save  # pylint: disable=invalid-name


@export
def load(file_path):
    """
    Load TBmodels objects from an HDF5 file.
    """
    try:
        return fsc.hdf5_io.load(file_path)
    except ValueError:
        with h5py.File(file_path, 'r') as hdf5_handle:
            return _legacy_decode._decode(hdf5_handle)  # pylint: disable=protected-access
