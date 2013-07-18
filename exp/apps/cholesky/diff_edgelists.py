#!/usr/bin/env python

"""diff_edgelists - Display differences between two edgelist files.

Uses a similarity threshold for the (floating-point) edge weight."""

import sys
from math import isnan

CLOSE_ENOUGH = 0.00001

def diff(fns, quick=False, verbose=False):
    """Display differences between edgelist files named in fns.
    If quick=True, stop after first difference.
    If verbose=True display each difference.
    Returns 0 if OK, maximal difference otherwise"""
    fhs = [open(i) for i in fns]
    errors = 0                  # Number of errors
    nedges = 0                  # Number of edges
    maxdiff = 0                 # Maximum observed difference
    # Iterate over the lines in each file (like diff(1) or paste(1) would do)
    for lines in zip(*(sorted(fh) for fh in fhs)):
        nedges += 1
        line = lines[0].strip().split()
        orig_nodes = [int(i) for i in line[0:2]]
        orig_data = [float(i) for i in line[2:]]
        mismatch = False
        for line in lines:
            line = line.strip().split()
            nodes = [int(i) for i in line[0:2]]
            data = [float(i) for i in line[2:]]
            for i in data:
                assert(not isnan(i))
            # Compute the error in the data
            werr = [abs(a-b) for a, b in zip(orig_data, data)]
            max_werr = max(werr)
            if orig_nodes != nodes or max_werr > CLOSE_ENOUGH:
                # FIXME: The numerics can be very different. This
                # doesn't seem to be due to the output format or
                # double/float differences between the Python version
                # and the C++ version.
                #
                # At the moment I think this is okay, since I just want to
                # make sure the answer is approximately correct, but 0.1
                # is getting a bit far from "approximately".
                mismatch = True
                if max_werr > maxdiff:
                    maxdiff = max_werr
        if mismatch:
            # Mismatch for this edge
            errors += 1
            if quick:
                break
            if verbose:
                for i, line in enumerate(lines):
                    print "[%d] %s" % (i, line.strip())
    if not quick:
        print "Checked %d edges, %d errors (max diff = %f)" % \
            (nedges, errors, maxdiff)
    return 0 if errors == 0 else maxdiff

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print "Usage: %s <file> <file> [...]"
        sys.exit(1)
    sys.exit(0 if diff(sys.argv[1:], verbose=True) == 0 else 1)
