#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    19.10.2015 00:08:40 CEST
# File:    read_model.py

from ._tb_model import Model
from ptools.locker import Locker
import ptools.sparse_matrix as sp

import re
import six
import numpy as np
import collections as co

#--------------------- COMMON HELPER FUNCTIONS -------------------------#
def _create_derivatives(fct):
    def from_file(path):
        with open(path, 'r') as f:
            string = f.read()
        return fct(string)
    globals()[fct.__name__ + '_file'] = from_file
    return fct

def _clean_input(string):
    r"""
    Removes comments, unnecessary whitespace and empty lines. Do not use for hr format (not insensitive to empty lines!).
    """
    lines = string.split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].split('#')[0]
    rx = re.compile('\s+')
    for i in range(len(lines)):
        lines[i] = rx.sub(' ', lines[i]).strip()
    return '\n'.join(list(filter(None, lines)))

def clean_string(fct):
    """
    Decorator for member functions taking (self, string), to clean string before any further processing.
    """
    def inner(self, string, *args, **kwargs):
        return fct(self, _clean_input(string), *args, **kwargs)
    return inner

#--------------------- TBM DATA FORMAT ---------------------------------#
@_create_derivatives
def tbm(string):
    """
    """
    return _tbm_read_impl(string).create_model()

@six.add_metaclass(Locker)
class _tbm_read_impl(object):
    def __init__(self, string):
        self.uc = None
        self.hop = None
        self.dim = None
        self.occ = None
        self.pos = None
        #~ self.size = None
        # set up all default inputs to Model
        self.sections = self._get_sections(string)
        sections_parsing_dict = co.OrderedDict(
            general=self._parse_general_section,
            uc=self._parse_uc_section,
            pos=self._parse_pos_section,
            hop=self._parse_hop_section,
        )
        # validate section names
        for key in self.sections.keys():
            if key not in sections_parsing_dict.keys():
                raise ValueError('Invalid section name \'{0}\'. Must be one of {1}'.format(key, section_parsing_dict.keys()))
        for key, fct in sections_parsing_dict.items():
            if key in self.sections.keys():
                fct(self.sections[key])

    @clean_string    
    def _get_sections(self, string, regex_str='\[([^\]]+)\]'):
        sct_name = re.compile(regex_str)
        sections_split = re.split(sct_name, string)
        if sections_split[0] != '':
            raise ValueError('Input \'{0}\' is not part of a section'.format(sections_split[0]))
        sections = dict()
        for i in range(1, len(sections_split), 2):
            name = sections_split[i].lower().strip()
            sections[name] = sections_split[i + 1]
        return sections

    @clean_string
    def _parse_uc_section(self, string):
        self.uc = self._parse_list(string)

    @clean_string
    def _parse_pos_section(self, string):
        self.pos = self._parse_list(string)

    @clean_string
    def _parse_list(self, string):
        return [
            [float(x) for x in line.split(' ')]
            for line in string.split('\n')
        ]

    # TODO: de-duplicate from _hop_list_model
    # This can be done when the different ways of setting up a Model
    # are put into functions.
    class _hop:
        """
        POD for hoppings
        """
        def __init__(self):
            self.data = []
            self.row_idx = []
            self.col_idx = []

        def append(self, data, row_idx, col_idx):
            self.data.append(data)
            self.row_idx.append(row_idx)
            self.col_idx.append(col_idx)

    @clean_string
    def _parse_hop_section(self, string):
        hop_sections = self._get_sections(string, '\(([^\)]+)\)')
        rx = re.compile('[\D]+')
        self.hop = dict()
        for key, val in hop_sections.items():
            key = list(filter(None, rx.split(key)))
            R = tuple([int(x) for x in key])
            self.hop[R] = self._parse_hop_lines(val)
            
    @clean_string
    def _parse_hop_lines(self, string):
        res = self._hop()
        rx = re.compile('[\s]+')
        for line in string.split('\n'):
            items = list(filter(None, rx.split(line)))
            i0 = int(items[0])
            i1 = int(items[1])
            t = complex(''.join(items[2:]))
            res.append(t, i0, i1)
        return sp.csr((res.data, (res.row_idx, res.col_idx)), dtype=complex, shape=(self.size, self.size))
        #~ return sp.csr((res.data, (res.row_idx, res.col_idx)), dtype=complex)
            
    @clean_string
    def _parse_general_section(self, string):
        # 'name': (type, Default)
        required_args = {
            'size': int,
        }
        optional_args = {
            'dim': (int, None),
            'occ': (int, None),
        }
        args_read = dict()
        rx = re.compile('[=:\s]+')
        for line in string.split('\n'):
            name, val = rx.split(line)
            name = name.lower().strip()
            args_read[name] = val

        for key, dtype in required_args.items():
            if key in args_read.keys():
                setattr(self, key, dtype(args_read[key]))
            else:
                raise ValueError('Missing required input argument {0} in [general] section.'.format(key))
        for key, (dtype, default) in optional_args.items():
            if key in args_read.keys():
                setattr(self, key, dtype(args_read[key]))
            else:
                setattr(self, key, default)

    def create_model(self):
        return Model(
            hop=self.hop,
            size=self.size,
            dim=self.dim,
            occ=self.occ,
            pos=self.pos,
            uc=self.uc,
            contains_cc=False
        )





#~ #--------------------- HR DATA FORMAT ----------------------------------#
#~ @_create_derivatives
#~ def hr(string):
    #~ """
    #~ """
    #~ raise NotImplemented
