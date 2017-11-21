#
# perf.py
#
# Performance number extractor for bmk2.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import re
import logging
log = logging.getLogger(__name__)

MULTIPLIERS = {'s': int(1E9), 'ms': int(1E6), 'us': int(1E3), 'ns': 1}

def split_decimal_str(n):
    p = n.find(".")
    if p != -1:
        whole = int(n[:p])
        frac = int(n[p+1:])
    else:
        whole = int(n)

    return (whole, frac)

class Perf(object):
    def get_perf(self, run):
        raise NotImplementedError

class ZeroPerf(object):
    def get_perf(self, run):
        return 0

class PerfFn(object):
    def __init__(self, fn):
        self.fn = fn

    def get_perf(self, run):
        if not (run.run_ok and run.check_ok):
            return None

        return self.fn(run.stdout, run.stderr)

class PerfRE(object):
    def __init__(self, rexp, re_unit = None):
        self.re = re.compile(rexp, re.MULTILINE)
        self.units = re_unit

        if re_unit:
            assert self.units in MULTIPLIERS, "Invalid unit %s" % (re_unit)

    def get_perf(self, run):
        if not (run.run_ok and run.check_ok):
            return None

        if run.stdout != None:
            run.stdout = run.stdout.replace("\r", "");
        if run.stderr != None:
            run.stderr = run.stderr.replace("\r", "");

        m = self.re.search(run.stdout)
        if not m and run.stderr:
            m = self.re.search(run.stderr)

        if not m:
            log.debug("No match for perf re in stdout or stderr")
            return None

        gd = m.groupdict()

        time_ns = 0
        if "time_ns" in gd:
            # use time_ns only if present
            time_ns = int(gd['time_ns'])
        elif "time_ms" in gd:
            time_ns = int(gd['time_ms']) * MULTIPLIERS['ms']
        elif "time_us" in gd:
            time_ns = int(gd['time_us']) * MULTIPLIERS['us']
        elif "time_s" in gd:
            time_ns = int(gd['time_s']) * MULTIPLIERS['s']
        elif "frac" in gd:
            w, f = int(gd['whole']), int(gd['frac'])

            assert self.units is not None

            m = MULTIPLIERS[self.units]

            l = len(str(m)) - len(gd['frac'])
            #print l
            assert l > 0, l

            time_ns = w * m + f * (10**(l-1))
        elif "float" in gd:
            assert self.units is not None

            m = MULTIPLIERS[self.units]
            
            time_ns = int(float(gd['float']) * m)
        else:
            assert False, "Unable to located named groups in perf regex (%s)" % (gd,)

        return {'time_ns': time_ns}


__all__ = ['Perf', 'ZeroPerf', 'PerfFn', 'PerfRE']
