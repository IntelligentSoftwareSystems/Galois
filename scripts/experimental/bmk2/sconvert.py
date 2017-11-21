#!/usr/bin/env python
#
# sconvert.py
#
# Simple converter for bmk2.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import convgraph
import re
import logging
import opdb
import argparse
import os

# simple converter

log = logging.getLogger(__name__)

class ConvSpec(opdb.ObjectPropsCFG):
    pass

def gen_xform_fn(srcname, dstname):
    src_re = re.compile(srcname)

    def f(s):
        return src_re.sub(dstname, s)

    return f

def load_convspec(convspec):
    cs = ConvSpec(convspec, "bmk2-convspec", ["2"])
    if not cs.load():
        log.error("Unable to read config file")
        return None

    return cs

def init_convgraph(cs):
    all_types = set()
    conv = {}

    for n, s in cs.objects.iteritems():    
        convgraph.register_conversion(s['src'], s['dst'], 
                                      gen_xform_fn(s['srcname'],
                                                   s['dstname']))

        all_types.add(s['src'])
        all_types.add(s['dst'])

        conv[(s['src'], s['dst'])] = n

    return all_types, conv


def convert_one(cs, src, srcty, dst, dstty, all_types, conv, exists = None, verbose = 0):
    if exists is None:
        exists = {}

    # might be useful for a copy?
    if dstty in exists:
        del exists[dstty]

    #print exists

    if srcty not in all_types:
        log.error("Conversion from %s not supported" % (srcty,))
        return None

    if dstty not in all_types:
        log.error("Conversion to %s not supported" % (dstty,))
        return None

    if dst == "@output":
        dst = None

    if not os.path.exists(src):
        log.error("Input file '%s' does not exist" % (src,))
        return None

    # silently skip destinations that already exist in database and on disk
    if dst and os.path.exists(dst):
        # we're also abandoning any intermediate files ...
        # TODO: the planner should do this...
        log.info("Destination `%s' already exists and is in database, not converting" % (dst,))
        return None
    
    c = convgraph.get_conversion(src, srcty, dst, dstty, exists, verbose)
    if not c:
        log.error("Unable to plan conversion from %s to %s" % (srcty, dstty))
        return None

    if False:
        print >>sys.stderr, c

    if dst is None:
        # we had to figure out the output name
        dst = c[-1][3]

    # skip destinations that only exist on disk but not in database
    if os.path.exists(dst):
        log.info("Destination `%s' already exists, not converting. But it is not in database, you need to update inputdb." % (dst,))
        return None

    out = []
    for cmd, fs, fsty, ds, dsty in c:
        assert cmd == "convert_direct", "Unsupported: %s" % (cmd,)
        assert (fsty, dsty) in conv, "Planner got it wrong: %s -> %s unsupported" % (fst, dsty)

        if os.path.exists(ds):
            continue

        cmd = cs.objects[conv[(fsty, dsty)]]['cmd']
        cmd = cmd.format(src = fs, dst=ds, verbose=1)

        out.append((ds, fs, cmd))

    return (dst, out)

def to_makefile(f, dst_rule_array):
    targets = []
    nout = []

    for dst, rules in dst_rule_array:
        targets.append(dst)

        for rule in rules:
            nout.append("""
{dst}: {src}
\t{cmd}""".format(src=rule[1], dst=rule[0], cmd=rule[2]))



    if len(targets):
        print >>f, "all: %s" % (" ".join(targets))
        print >>f, "\n".join(nout)

if __name__ == '__main__':
    import bmk2
    import config
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(name)-10s %(message)s')

    p = argparse.ArgumentParser(description="Convert commands to convert input file to destination")
    p.add_argument("input", help="Input file")
    p.add_argument("input_type", help="Input file type")
    p.add_argument("dst_type", help="Destination file type (name will be autodetermined)")
    p.add_argument("dst", nargs="?", help="Destination file name (optional)")

    p.add_argument("-o", dest="output", metavar="FILE", help="Output makefile", default="/dev/stdout")
    p.add_argument("-d", dest="metadir", metavar="PATH", help="Path to load configuration from", default=".")
    p.add_argument("-v", dest="verbose", type=int, help="Verbosity", default=0)

    args = p.parse_args()
    
    cfg = config.Config(args.metadir)
    if not cfg:
        sys.exit(1)

    convspec = cfg.get_var('convspec', None)
    if not convspec:
        log.error("No 'convspec' in config file")
        sys.exit(1)

    cs = load_convspec(os.path.join(cfg.metadir, convspec))
    if not cs:
        sys.exit(1)
        
    all_types, conv = init_convgraph(cs)
    cmds = convert_one(cs, args.input, args.input_type, args.dst, args.dst_type, all_types, conv)
    if cmds is None:
        sys.exit(1)

    f = open(args.output, "w")
    to_makefile(f, [cmds])
    f.close()
