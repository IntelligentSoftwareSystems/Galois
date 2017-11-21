#
# extras.py
#
# Utility functions for bmk2.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import os
import fnmatch

def read_line_terminated_cfg(configs):
    out = []
    for f in configs:
        fl = [s.strip() for s in open(f, "r")]
        fl = [l for l in fl if (l and l[0] != "#")]
        out += fl

    return out

# Tyler: Not sure if this is the best place for the blacklist
def scan(path, glob, black_list = []):
    out = []
    for root, dirnames, filenames in os.walk(path):
        matches = fnmatch.filter(filenames, glob)
        out += [os.path.join(root, m) for m in matches]

    out = [o for o in out if len([x for x in black_list if x + os.sep in o]) == 0]
    return out

def summarize(log, rspecs):
    bins = set([rs.bmk_binary.get_id() for rs in rspecs])
    inputs = set([rs.bmk_input.get_id() for rs in rspecs])

    runs = 0
    failed_runs = 0
    failed_checks = 0

    for rs in rspecs:
        runs += len(rs.runs)
        failed_runs += len(filter(lambda x: not x.run_ok, rs.runs))
        failed_checks += len(filter(lambda x: not x.check_ok, rs.runs))

    log.info('Summary: Runspecs: %s Binaries: %d Inputs: %d  Total runs: %d Failed: %d Failed Checks: %d' % (len(rspecs), len(bins), len(inputs), runs, failed_runs, failed_checks))

def standard_loader(metadir, inpproc, binspec, scandir, bispec, binputs = "", 
                    ignore_missing_binaries = False, bingroup = "BINARIES", 
                    bin_configs = None, extended_scan = False, black_list = [], 
                    varconfigs = None):
    import bmk2
    import config
    import sys

    if scandir:
        basepath = os.path.abspath(scandir)
        binspecs = scan(scandir, "bmktest2.py", black_list)
        if extended_scan:
            binspecs.extend(scan(scandir, "bmktest2-*.py", black_list))
    else:
        if not os.path.exists(binspec):
            print >>sys.stderr, "Unable to find %s" % (binspec,)
            return False

        basepath = os.path.abspath(".")
        binspecs = [binspec]

    l = bmk2.Loader(metadir, inpproc)

    ftf = {}
    if bispec:
        f = None
        if os.path.exists(bispec) and os.path.isfile(bispec):
            f = bispec
        else:
            f = l.config.get_var("bispec_" + bispec, None)
            f = os.path.join(metadir, f)

        assert f is not None, "Unable to find file or spec in config file for bispec '%s'" % (bispec,)
        ftf[config.FT_BISPEC] = f

    if not l.initialize(ftf): return False
    sel_inputs, sel_binaries = l.split_binputs(binputs)

    print >>sys.stderr, "sel_inputs set to '%s', sel_binaries set to '%s'" % (sel_inputs, sel_binaries)

    if bin_configs is not None and len(bin_configs) > 0:
        if not l.config.load_bin_config(bin_configs):
            print >>sys.stderr, "Unable to load binary configurations '%s'" % (bin_configs,)
            return False

    if varconfigs is not None:
        if not l.config.load_var_config(varconfigs):
            print >>sys.stderr, "Unable to load variable configurations '%s'" % (varconfigs,)
            return False

    sys.path.append(metadir)
    if not l.load_multiple_binaries(binspecs, sel_binaries, bingroup) and not ignore_missing_binaries: return False
    if not l.apply_config(): return False
    if not l.associate_inputs(sel_inputs): return False

    return (basepath, binspecs, l)

if __name__ == '__main__':
    import sys
    print scan(sys.argv[1], "bmktest2.py")
