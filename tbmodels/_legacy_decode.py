from ._tb_model import Model


def _decode(hf):
    if 'tb_model' in hf or 'hop' in hf:
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
