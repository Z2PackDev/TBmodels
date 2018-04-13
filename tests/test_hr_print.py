#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile

import pytest

import tbmodels
import numpy as np

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('t', T_VALUES)
def test_hr_print(t, get_model, compare_equal):
    model = get_model(*t)
    compare_equal(model.to_hr().splitlines()[1:])  # timestamp in first line isn't equal


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency(hr_name, sample):
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_hr_file(hr_file, occ=28, sparse=True)
    lines_new = model.to_hr().split('\n')
    with open(hr_file, 'r') as f:
        lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
    assert len(lines_new) == len(lines_old)
    for l_new, l_old in zip(lines_new[1:], lines_old[1:]):
        assert l_new.replace('-0.00000000000000',
                             ' 0.00000000000000') == l_old.replace('-0.00000000000000', ' 0.00000000000000')


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency_file(hr_name, models_equal, sparse, sample):
    hr_file = sample(hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, sparse=sparse)
    with tempfile.NamedTemporaryFile() as tmpf:
        model1.to_hr_file(tmpf.name)
        model2 = tbmodels.Model.from_hr_file(tmpf.name, sparse=sparse)
    models_equal(model1, model2)


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency_str(hr_name, models_equal, sparse, sample):
    hr_file = sample(hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, sparse=sparse)
    model2 = tbmodels.Model.from_hr(model1.to_hr(), sparse=sparse)
    models_equal(model1, model2)


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'hr_hamilton_full.dat'])
def test_consistency_no_hcutoff(hr_name, sample):
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_hr_file(hr_file, occ=28, h_cutoff=-1, sparse=True)
    lines_new = model.to_hr().split('\n')
    with open(hr_file, 'r') as f:
        lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
    assert len(lines_new) == len(lines_old)
    for l_new, l_old in zip(lines_new[1:], lines_old[1:]):
        assert l_new.replace('-0.00000000000000',
                             ' 0.00000000000000') == l_old.replace('-0.00000000000000', ' 0.00000000000000')


def test_invalid_empty():
    model = tbmodels.Model(size=2, dim=3)
    with pytest.raises(ValueError):
        model.to_hr()
