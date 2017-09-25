#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess

import tbmodels as tb

if __name__ == '__main__':
    WANNIER90_COMMAND = os.path.expanduser(
        '~/programming/wannier90/wannier90.x'
    )
    BUILD_DIR = './build'

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.copytree('./input', BUILD_DIR)
    subprocess.call([WANNIER90_COMMAND, 'silicon'], cwd=BUILD_DIR)

    model = tb.Model.from_wannier_folder(BUILD_DIR, prefix='silicon')
    print(model.eigenval([0, 0, 0]))
