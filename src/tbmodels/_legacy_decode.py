#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines decoding for the legacy (pre fsc.hdf5-io) HDF5 format.
"""

from ._tb_model import Model


def _decode(hdf5_handle):
    """
    Decode the object at the given HDF5 node.
    """
    if "tb_model" in hdf5_handle or "hop" in hdf5_handle:
        return _decode_model(hdf5_handle)
    elif "val" in hdf5_handle:
        return _decode_val(hdf5_handle)
    elif "0" in hdf5_handle:
        return _decode_iterable(hdf5_handle)
    else:
        raise ValueError("File structure not understood.")


def _decode_iterable(hdf5_handle):
    return [_decode(hdf5_handle[key]) for key in sorted(hdf5_handle, key=int)]


def _decode_model(hdf5_handle):
    return Model.from_hdf5(hdf5_handle)


def _decode_val(hdf5_handle):
    return hdf5_handle["val"][()]
