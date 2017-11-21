#!/usr/bin/env python
# measure_energy.py
#
# Measure energy on Intel platforms that support RAPL access through
# the powercap interface.
#
# Part of bmk2
#
# Copyright (c) 2017 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>

import sys
import os
import subprocess
import glob
import threading

INTERVAL_S = 15

def get_rapl_files():
    dom = glob.glob("/sys/class/powercap/intel-rapl:*")
    
    out = {}
    for d in dom:
        dd = os.path.basename(d)
        f = os.path.join(d, "energy_uj")
        if os.path.exists(f):
            out[dd] = f

        f = os.path.join(d, "max_energy_range_uj")
        if os.path.exists(f):
            out["max_" + dd] = f


    return out

def read_rapl_power(rapl_files):
    out = {}
    for k, f in rapl_files.items():
        of = open(f, "r")
        out[k] = int(of.read())
        of.close()

    return out

def periodic_power():
    global TIMER
    POWER.append(read_rapl_power(rf))
    TIMER = threading.Timer(INTERVAL_S, periodic_power)
    TIMER.start()

def count_wraparound(nums, key = None):
    prev = None
    wrap = 0

    for n in nums:
        if key:
            n = key(n)

        if prev is not None and prev > n:
            wrap += 1

        prev = n

    return wrap

def calc_power(POWER):
    k = [kk for kk in POWER[0].keys() if kk[:4] != 'max_']

    out = {}
    for kk in k:
        wraps = count_wraparound(POWER, key = lambda x: x[kk])
        
        bef = POWER[0][kk]
        aft = POWER[-1][kk] + wraps * POWER[-1]["max_" + kk]

        out[kk] = aft - bef
        out[kk+":wraps"] = wraps

    return out

if len(sys.argv) == 1:
    print >>sys.stderr, "Usage: %s cmd-line\n" % (sys.argv[0],)
    exit(1)

cmdline = sys.argv[1:]

rf = get_rapl_files()
if len(rf):
    POWER = []
    TIMER = threading.Timer(INTERVAL_S, periodic_power)
    POWER.append(read_rapl_power(rf))
    TIMER.start()

    proc = subprocess.Popen(cmdline)
    proc.wait()

    TIMER.cancel()
    POWER.append(read_rapl_power(rf))

    p = calc_power(POWER)
    for k in p:
        print "INSTR", k, p[k] # micro joules

    sys.exit(proc.returncode)
else:
    print >>sys.stderr, "Did not find RAPL power counters (/sys/class/powercap/intel-rapl*)"
    sys.exit(1)
