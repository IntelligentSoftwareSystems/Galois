#!/usr/bin/env python
#
# convgraph.py
#
# Planner-based graph converter library.  
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

from pyhop import *

conversions = {}

def register_conversion(src, dst, fn_xform):
    assert (src, dst) not in conversions, "Duplicate conversion (%s, %s)" % (src, dst)

    conversions[(src, dst)] = fn_xform

def convert_direct(state, a, fmt_a, b, fmt_b):
    # we must have a direct converter
    if (fmt_a, fmt_b) not in conversions:
        #print "no direct conversion"
        return False

    if state.files[fmt_a] != a and state.existing[fmt_a] != a:
        #print "src does not exist"
        return False

    if state.files[fmt_b] == b:
        #print "dst exists"
        return False
    
    state.files[fmt_b] = b
    return state

declare_operators(convert_direct)

def convert_from_existing(state, a, fmt_a, b, fmt_b):
    if (fmt_a, fmt_b) in conversions:        
        if b is None:
            b = conversions[(fmt_a, fmt_b)](a)
            if b == a:
                return False

        return [('convert_direct', a, fmt_a, b, fmt_b)]

    for f, e in state.existing.iteritems():
        if state.files[f] is None and f not in state.tried_existing:
            state.tried_existing.add(f)
            # exists but does not feature as a step
            return [('convert', e, f, b, fmt_b)]

    return False

def convert_via(state, a, fmt_a, b, fmt_b):
    if (fmt_a, fmt_b) in conversions:
        return False
    else:
        for s, d in conversions:
            if s == fmt_a and state.files[d] is None:
                via = conversions[(s, d)](a)

                if via == a: # did not match regex
                    return False

                return [('convert', a, fmt_a, via, d),
                        ('convert', via, d, b, fmt_b)]
        
        return False

declare_methods('convert', convert_from_existing, convert_via)

def get_conversion(start, start_ty, end, end_ty, existing, verbose=0):
    s = State('initial')
    
    # do we need existing?
    s.existing = {}
    s.files = {}
    s.tried_existing = set()

    for f1, f2 in conversions.keys():
        s.files[f1] = None
        s.files[f2] = None

    for k, v in existing.iteritems():
        s.existing[k] = v

    s.files[start_ty] = start

    x = pyhop(s, [('convert', start, start_ty, end, end_ty)], verbose=verbose)
    return x

if __name__ == "__main__":
    start_file = 'a'
    start_file_fmt = 'binary/gr'

    s = State('initial')
    s.existing = {}
    s.files = {}

    for f1, f2 in conversions.keys():
        s.files[f1] = None
        s.files[f2] = None

    s.files[start_file_fmt] = start_file

    s.existing[start_file_fmt] = start_file
    s.existing['other/format1'] = 'c'

    x = pyhop(s, [('convert', 'a', 'binary/gr', 'b', 'other/format')], verbose=2)
    if not x:
        print "conversion is unsupported"

        
