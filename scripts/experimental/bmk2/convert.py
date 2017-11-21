#!/usr/bin/env python
#
# convert.py
#
# Bulk converter for graph files in bmk2. 
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import sys
import ConfigParser
import argparse
from extras import *
import logging
import opdb
import os
import re
import sconvert

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(name)-10s %(message)s')

p = argparse.ArgumentParser("Generate conversion makefile")
p.add_argument("output", nargs="?", default="/dev/stdout")
p.add_argument("-d", dest="metadir", metavar="PATH", help="Path to load configuration from", default=".")
p.add_argument("--iproc", dest="inpproc", metavar="FILE", help="Input processor")
p.add_argument("--bs", dest="binspec", metavar="FILE", help="Binary specification", default="./bmktest2.py")
p.add_argument("--bispec", dest="bispec", metavar="FILE_OR_MNEMONIC", help="Binary+Input specification")
p.add_argument("--scan", dest="scan", metavar="PATH", help="Recursively search PATH for bmktest2.py")
p.add_argument("-v", dest="verbose", type=int, help="Verbosity", default=0)

args = p.parse_args()

loaded = standard_loader(args.metadir, args.inpproc, args.binspec, args.scan, args.bispec, bingroup='CONVERTERS')
if not loaded:
    sys.exit(1)
else:
    basepath, binspecs, l = loaded

convspec = l.config.get_var('convspec', None)
if not convspec:
    log.error("No 'convspec' in config file")
    sys.exit(1)

cs = sconvert.load_convspec(os.path.join(l.config.metadir, convspec))
if not cs:
    sys.exit(1)

all_types, conv = sconvert.init_convgraph(cs)

out = []
rspecs = l.get_run_specs()
for rs in rspecs:
    src, srcty, dst, dstty = rs.args
    src, srcty, dst, dstty = src[0], srcty[0], dst[0], dstty[0]

    exists = {}
    for alt in rs.bmk_input.get_all_alt():
        if alt.props.format not in all_types:
            log.error("Format '%s' not listed in convspec"%  (alt.props.format,))
            sys.exit(1)

        if os.path.exists(alt.props.file):
            # sometimes alt.props.file may only exist in the database
            exists[alt.props.format] = alt.props.file
            
    cmds = sconvert.convert_one(cs, src, srcty, dst, dstty, all_types, conv, exists, args.verbose)
    if cmds is None:
        continue

    out.append(cmds)

if len(out):
    f = open(args.output, "w")
    sconvert.to_makefile(f, out)
    f.close()
