#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:29:31 2023

@author: matthewromer
"""
import os
import platform

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

def clear_make_dir(dir):
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        run_cmd('rm -rf {}'.format(dir))
        run_cmd('mkdir {}'.format(dir))
    elif platform.system() == 'Windows':
        run_cmd('rmdir /Q /S  {}'.format(dir))
        run_cmd('mkdir  {}'.format(dir))
    else:
        raise ValueError('Platform `{}` not supported'.format(platform.system()))