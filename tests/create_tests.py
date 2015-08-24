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
    arguments = parser.parse_args(sys.argv[1:])
    #~ sys.argv = [sys.argv[0]] # turns off forwarding the flags

    exclude_list = {}

    for filename in os.listdir('./templates'):
        try:
            if not exclude_list[filename.split('.')[0]]:
                continue
        except KeyError:
            pass
        shutil.copyfile('./templates/' + filename, './' + filename)
    execfile('test.py', globals(), locals())
    
