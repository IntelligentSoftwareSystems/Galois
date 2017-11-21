#!/usr/bin/env python
#
# inputdb.py
#
# Manages input db files for bmk2 (*.inputdb)
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import sys
import os
import ConfigParser
import argparse
import common
import fnmatch
import inputprops
from core import Input
from opdb import ObjectPropsCFG

class InputDBcfg(ObjectPropsCFG):
    """Parser of inputdb files."""
    def __init__(self, filename, inpproc = None):
        super(InputDBcfg, self).__init__(filename, "bmktest2", ["2"])
        self.inpproc = inpproc

        self.unserialize_input = None
        self.serialize_input = None

        # save input processor functions if they exist
        if self.inpproc:
            inpproc = common.load_py_module(self.inpproc)
            if 'unserialize_input' in inpproc:
                self.unserialize_input = inpproc['unserialize_input']

            if 'serialize_input' in inpproc:
                self.serialize_input = inpproc['serialize_input']                

    def init(self, basepath):
        self.meta = dict([('version', "2"), ('basepath', basepath)])

    def post_load(self):
        """Append the basepath to file paths of all "file" variables
        loaded. (basepath specified in meta section)
        """
        basepath = os.path.expanduser(self.meta['basepath'])

        if not (os.path.exists(basepath)):
            print >>sys.stderr, "Basepath '%s' ('%s') does not exist" % (basepath, self.meta['basepath'])
            return False

        for s in self.objects:
            e = self.objects[s]
            if self.unserialize_input:
                e = self.unserialize_input(e, basepath)

            e['file'] = os.path.join(basepath, e['file'])

        return True
        #basepath = self.meta['basepath']

        #for s in self.objects:
        #    e = self.objects[s]
        #    if self.unserialize_input:
        #        e = self.unserialize_input(e, basepath)

        #    e['file'] = os.path.join(basepath, e['file'])            

        #return True

    def unparse_section(self, section):
        if self.serialize_input:
            self.serialize_input(section)
            
        # replaces the "file" variable in a section with a RELATIVE path
        #if 'file' in section:
        #    section['file'] = os.path.relpath(section['file'], self.meta['basepath'])
        if 'file' in section:
            basepath = os.path.expanduser(self.meta['basepath'])
            section['file'] = os.path.relpath(section['file'], basepath)

        return section

class InputDB(object):
    """Database of inputs specified by an inputDB."""
    def __init__(self, cfgfile, inpproc = None, inputprops = None):
        self.inpproc = inpproc
        self.inputprops = inputprops
        self.cfg = InputDBcfg(cfgfile, self.inpproc)

    def get_alt_format(self, name, fmt):
        """Get the Input object associated with some input that has a 
        particular format.
        """
        if name in self.n2i:
            for x in self.n2i[name]:
                if x.props.format == fmt:
                    return x

    def get_all_alt(self, name):
        """Get the Input object(s) associated with some input."""
        if name in self.n2i:
            return self.n2i[name]

    def load(self):
        if not self.cfg.load():
            print >>sys.stderr, "Unable to load InputDB configuration!"
            return False

        if self.inputprops is not None:
            # not .props as Properties!
            self.props = inputprops.InputPropsCfg(self.inputprops, self)
            if not self.props.load():
                print >>sys.stderr, "Unable to load InputProps"
                return False

            # add any new properties specified by inputprops into the
            # correct inputs
            inputprops.apply_props(self.cfg.objects.itervalues(), self.props)

        # get all inputs specified by inputdb and wrap into an Input
        # object
        self.inputdb = [Input(i, self) for i in self.cfg]
        self.inpnames = set([i.get_id() for i in self.inputdb])

        # setup the name -> Input map; note an input can have multiple
        # formats...
        self.n2i = dict([(n, list()) for n in self.inpnames])
        for i in self.inputdb:
            self.n2i[i.get_id()].append(i)
        
        return True
            
    def __iter__(self):
        return iter(self.inputdb)
       
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare an inputs database")
    p.add_argument("--glob", help="Glob")
    p.add_argument("--update", action="store_true", help="Update dbfile")
    p.add_argument("inpproc", help="Input processor (python module)")
    p.add_argument("dbfile", help="Output database file")
    p.add_argument("basepath", nargs="?", help="Scan this path for inputs", default=".")
    
    args = p.parse_args()
    inpproc = common.load_py_module(args.inpproc)

    if args.update:
        idb = InputDB(args.dbfile, args.inpproc)
        idb.load()
        basepath = os.path.expanduser(idb.cfg.meta['basepath'])
        print >>sys.stderr, "using basepath from file: %s" % (basepath,)
    else:
        idb = InputDB(args.dbfile, args.inpproc)
        basepath = args.basepath
        idb.cfg.init(basepath)

    describe_input = inpproc['describe_input']

    out = []
    for root, dirnames, filenames in os.walk(basepath, followlinks=True):
        rp = os.path.relpath(root, basepath)
        
        if args.glob:
            filenames = fnmatch.filter(filenames, args.glob)       

        for f in filenames:
            if f[0] == ".":
                continue

            x = describe_input(root, f, rp)
            if x:
                x['file'] = os.path.join(rp, f)
                if x['file'] not in idb.cfg.objects:
                    print >>sys.stderr, x['file']
                    idb.cfg.objects[x['file']] = x
                    x['file'] = os.path.join(basepath, x['file'])
                    
    if args.update:
        idb.cfg.save(args.dbfile)
    else:
        idb.cfg.save(args.dbfile)

