#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    31.10.2014 09:53:11 CET
# File:    create_tests.py

from common import *
import sys

import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', dest='name')
    arguments = parser.parse_args(sys.argv[1:])
    #~ sys.argv = [sys.argv[0]] # turns off forwarding the flags

    if arguments.name is not None:
        filename = arguments.name
        shutil.copyfile('./templates/' + filename, './' + filename)
        sys.argv = [sys.argv[0]]
        execfile(filename, globals(), locals())
    else:
        exclude_list = {}

        for filename in os.listdir('./templates'):
            try:
                if not exclude_list[filename.split('.')[0]]:
                    continue
            except KeyError:
                pass
            shutil.copyfile('./templates/' + filename, './' + filename)
        execfile('test.py', globals(), locals())
