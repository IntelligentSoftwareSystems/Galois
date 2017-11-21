#
# common.py
#
# Python utilities for bmk2.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

# runs a python file and returns the globals
def load_py_module(f):
    """Executes a python file and returns the globals in it at the end
    of execution.
    """
    g = {}
    x = execfile(f, g)
    return g
