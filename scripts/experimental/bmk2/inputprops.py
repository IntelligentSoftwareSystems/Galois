#!/usr/bin/env python
#
# inputprops.py
#
# Manages an input properties file (*.inputprops)
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import sys
import inputdb
import argparse
import ConfigParser
import os
from opdb import ObjectPropsCFG

class InputPropsCfg(ObjectPropsCFG):
    """Parser of inputprops files that specify additional properties
    for certain inputs.
    """
    def __init__(self, filename, inputdb):
        super(InputPropsCfg, self).__init__(filename, "bmktest2-props", ["2"])
        self.inputdb = inputdb
        self.path_items = set()

    def init(self):
        self.meta = {}
        self.meta['version'] = "2"

    def post_load(self):
        """Look for "path_items" as a var name in the sections; if it exists,
        then prepend the base path (specified in meta of inputdatabase) to the 
        path.
        """
        path_items = self.meta.get("paths", "")
        self.path_items = set([xx.strip() for xx in path_items.split(",")])

        basepath = os.path.expanduser(self.inputdb.cfg.meta['basepath'])

        for e in self.objects.itervalues():
            for pi in self.path_items:
                if pi in e:
                    e[pi] = os.path.join(basepath, e[pi])

        return True

    def unparse_section(self, section):
        """Replace anything in path_items with relative paths."""
        bp = os.path.expanduser(self.inputdb.cfg.meta['basepath'])

        for pi in self.path_items:
            if pi in section:
                section[pi] = os.path.relpath(section[pi], bp)

        return section

def apply_props(inputdb, props):
    """Save the additional properties specified by an inputprops file
    into the input database.
    """
    for e in inputdb:
        if e['name'] in props.objects:
            e.update(props.objects[e['name']])

    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser("Create/Update an input properties file")
    p.add_argument("inputdb", help="Inputdb file")
    p.add_argument("inputprops", help="Inputprops file")

    args = p.parse_args()

    idb = inputdb.InputDB(args.inputdb)
    ip = InputPropsCfg(args.inputprops, idb)

    
    if not idb.load():
        print >>sys.stderr, "Failed to load inputdb"
        sys.exit(1)


    if os.path.exists(args.inputprops):
        if not ip.load():
            print >>sys.stderr, "Failed to load props"
            sys.exit(1)
    else:
        ip.init()

    for e in idb:
        nm = e.name

        if nm not in ip.objects:
            ip.objects[nm] = {'name':  nm}

    ip.save(args.inputprops)
