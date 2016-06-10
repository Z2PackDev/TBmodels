#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 12:04:36 CEST
# File:    _hr_hamilton.py

from __future__ import division

import numpy as np

from ._hop_list_model import HopListModel

class HrModel(HopListModel):
    r"""A subclass of :class:`tb.Model` designed to read the
    tight-binding model from the ``*_hr.dat`` file produced by Wannier90.

    :param hr_file: Path to the ``*_hr.dat`` file.
    :type hr_file: str

    :param h_cutoff: Minimum absolute value for hopping parameters to
        be included. This is useful if the ``hr_file`` contains many
        zero entries. Default: ``None`` (no hopping entries are excluded).
    :type h_cutoff: float

    :param kwargs: Keyword arguments are passed to :class:`Model` . For ``add_cc``, the default is ``False`` (unlike in :class:`HopListModel`).
    """
    def __init__(self, hr_file, h_cutoff=None, **kwargs):
        with open(hr_file, 'r') as file_handle:
            num_wann, h_entries = _read_hr(file_handle)
            if h_cutoff is not None:
                h_entries = [hopping for hopping in h_entries if abs(hopping[3]) > h_cutoff]

            if 'add_cc' not in kwargs.keys():
                kwargs['add_cc'] = False
            super(HrModel, self).__init__(size=num_wann, hop_list=h_entries, **kwargs)

def _read_hr(file_handle):
    r"""
    read the number of wannier functions and the hopping entries
    from *hr.dat and converts them into the right format
    """
    file_handle = enumerate(file_handle)
    next(file_handle) # skip first line
    num_wann = int(next(file_handle)[1])
    nrpts = int(next(file_handle)[1])

    # get degeneracy points
    deg_pts = []
    # order in zip important because else the next data element is consumed
    for _, (_, line) in zip(range(int(np.ceil(nrpts / 15))), file_handle):
        deg_pts.extend(int(x) for x in line.split())
    assert len(deg_pts) == nrpts

    num_wann_square = num_wann**2
    def to_entry(line, i):
        """Turns a line (string) into a hop_list entry"""
        line_no, line = line
        entry = line.split()
        orbital_a = int(entry[3]) - 1
        orbital_b = int(entry[4]) - 1
        # test consistency of orbital numbers
        if not (orbital_a == i % num_wann and orbital_b == (i % num_wann_square) // num_wann):
            raise ValueError('Inconsistent orbital numbers in line number {}'.format(line_no + 1))
        return [
            orbital_a,
            orbital_b,
            [int(x) for x in entry[:3]],
            (float(entry[5]) + 1j * float(entry[6])) / (deg_pts[i // num_wann_square])
        ]

    # skip random empty lines
    lines_nonempty = (l for l in file_handle if l[1].strip())
    hop_list = (to_entry(line, i) for i, line in enumerate(lines_nonempty))

    return num_wann, hop_list
