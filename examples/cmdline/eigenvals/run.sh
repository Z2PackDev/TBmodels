#!/bin/bash
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

tbmodels eigenvals -i input/silicon_model.hdf5 -k input/kpoints.hdf5 -o silicon_bands.hdf5
