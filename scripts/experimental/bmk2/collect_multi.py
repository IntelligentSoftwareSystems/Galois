#!/usr/bin/env python
#
# collect_multi.py
#
# Scans multiple log files for "COLLECT" and outputs a list of files to be
# collected. Part of bmk2.
#
# Copyright (c) 2015, 2016, 2017 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

from collect import *
import argparse
import mapfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect extra files generated during test2.py in a single directory")
    parser.add_argument('logfiles', nargs='+', help='Logfiles')
    parser.add_argument('-t', dest="filetype", action="append", help='Type of files to collect (default: all)', default=[])
    parser.add_argument('-p', dest="strip_path", type=int, metavar='NUM', help='Strip NUM components from filename before combining with basepath', default=0)
    parser.add_argument('-m', dest="map", metavar='FILE', help='Store map of RSID, file and filetype in FILE', default=None)
    parser.add_argument("-a", dest='append', action='store_true', default=False, help="Append to map file")
    parser.add_argument('-s', dest="suffix", metavar='SUFFIX', help='Add suffix to filename', default=0)
    parser.add_argument('--collect-failed', dest="skip_failed", action="store_false", default=True, help='Collect files from failed runs')

    args = parser.parse_args()

    ft = set(args.filetype)

    for i, l in enumerate(args.logfiles):
        out, fnames, revmap = collect_logfile(l, args.skip_failed, args.strip_path, args.suffix, ft)
        print "\n".join(out)

        if args.map:
            mapfile.write_mapfile_raw(args.map, mapentries(fnames, revmap), "a" if (args.append or i > 0) else "w")
