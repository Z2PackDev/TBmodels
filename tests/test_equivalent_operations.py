"""
Test that equivalent operations produce the same model.
"""

import tbmodels


def test_shift_supercell(sample, models_close):
    """
    Test that shifting the unit cell is equivalent to creating and
    then folding a supercell.
    """
    model = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    shift = model.uc[2] / 2 - 0.03

    model_shifted = model.change_unit_cell(offset=shift, cartesian=True)

    supercell_model = model.supercell(size=(1, 1, 2))
    model_folded = supercell_model.fold_model(
        new_unit_cell=model.uc, unit_cell_offset=shift, orbital_labels=list(range(model.size)) * 2
    )

    assert models_close(model_shifted, model_folded)
