#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

"""
This script parses a tight-binding model from Wannier90 output and creates the corresponding HDF5 file.
"""

import os
import sys
import tbmodels

usage = "./parse_model.py <input_folder/prefix> <output_name>"

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(usage)
        sys.exit()

    common_path = sys.argv[1]
    input_files = dict()
    input_files['hr_file'] = common_path + '_hr.dat'

    for key, suffix in [
            ('win_file', '.win'),
            ('wsvec_file', '_wsvec.dat'),
            ('xyz_file', '_centres.xyz'),
    ]:
        filename = common_path + suffix
        if os.path.isfile(filename):
            input_files[key] = filename

    model = tbmodels.Model.from_wannier_files(ignore_orbital_order=True, **input_files)
    model.to_hdf5_file(sys.argv[2] + '.hdf5')
