import h5py
import fsc.hdf5_io
from fsc.export import export

from . import _legacy_decode

__all__ = ['save']

save = fsc.hdf5_io.save


@export
def load(file_path):
    try:
        return fsc.hdf5_io.load(file_path)
    except ValueError:
        with h5py.File(file_path, 'r') as hf:
            return _legacy_decode._decode(hf)
