#
# bmk2.py
#
# Loader for bmk2 tests.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import sys
from common import *
from config import *
import glob
import os
import inputdb
import bispec
from core import *
from checkers import *
from perf import *

import logging
log = logging.getLogger(__name__)

def load_binary_specs(f, binary_group = 'BINARIES'):
    """Load a python file which should have a global variable
    which contains a list of test specifications to run.
    """
    g = load_py_module(f)

    if binary_group in g:
        return g[binary_group]
    else:
        log.error("No %s in %s" % (binary_group, f))
        return None
       
class Loader(object):
    def __init__(self, metadir, inpproc):
        self.config = Config(metadir, inpproc)        
        self.binaries = {}
        self.bin_inputs = {}
        self.inp_filtered = False

    def initialize(self, ftf = {}):
        """Load the input database, processor, and properties as well as
        the binary spec which specifies which inputs should be run with
        certain binaries.
        """
        # load the configuration files
        if not self.config.load_config():
            return False
        
        if not self.config.auto_set_files():
            return False

        for ty, f in ftf.iteritems():
            if isinstance(f, list):
                for ff in f:
                    self.config.set_file(ff, ty)
            else:
                self.config.set_file(f, ty)

        # load the input database from specified config files
        self.inputdb = inputdb.InputDB(self.config.get_file(FT_INPUTDB), 
                                       self.config.get_file(FT_INPUTPROC),
                                       self.config.get_file(FT_INPUTPROPS))
        if not self.inputdb.load():
            return False

        # load the binary -> input mapping
        self.bs = bispec.read_bin_input_spec(self.config.get_file(FT_BISPEC))
        self.bs.set_input_db(self.inputdb)

        return True

    def split_binputs(self, binputs):
        bins = set()
        inputs = set()

        if binputs:
            inpnames = self.inputdb.inpnames

            for i in binputs:
                if i in inpnames:
                    inputs.add(i)
                else:
                    bins.add(i)

        self.inp_filtered = len(inputs) > 0

        return inputs, bins            

    def load_multiple_binaries(self, binspecs, sel_binaries = None, bingroup = "BINARIES"):
        for b in binspecs:
            if not self.load_binaries(b, sel_binaries, bingroup):
                return False

        return True

    def load_binaries(self, binspec, sel_binaries = None, bingroup = "BINARIES"):
        """Load the list of binaries that need to be run from a python file
        with a global object containing the names of Binary classes.
        """
        d = os.path.dirname(binspec)
        binaries = load_binary_specs(binspec, bingroup)
        if binaries:
            for b in binaries:
                if b.get_id() in self.binaries:
                    log.error("Duplicate binary id %s in %s" % (b.get_id(), binspec))
                    return False

                if sel_binaries and b.get_id() not in sel_binaries:
                    log.debug("Ignoring binary id %s in %s, not in sel_binaries" % (b.get_id(), binspec))
                    continue

                self.binaries[b.get_id()] = b

                if d == '':
                    log.warning('binspec path from "%s" is empty' % (binspec,))

                b.props._cwd = d

            return True
        
        if not binaries or len(binaries) == 0:
            log.error("%s is empty in %s" % (bingroup, binspec))

        return False

    def apply_config(self):
        """TODO figure out what this does"""
        if len(self.binaries) == 0:
            log.error("No binaries to apply configuration to.")
            return False

        if self.config.bin_config is not None and (self.config.bin_config):
            log.info('Applying configuration "%s"' % (self.config.bin_config,))

            for b in self.binaries.itervalues():
                b.apply_config(self.config.bin_config)
        else:
            log.info('No binary-specific configurations specified')
        
        return True

    def associate_inputs(self, binputs = None):
        """Given loaded binary inputs + binaries, associate inputs with
        binaries.
        """
        if len(self.binaries) == 0:
            log.error("No binaries")
            return False

        for bid, b in self.binaries.iteritems():
            i = self.bs.get_inputs(b, binputs)
            if len(i) == 0:
                if not self.inp_filtered:
                    log.error("No inputs matched for binary " + bid)
                    return False
                else:
                    log.warning("No inputs matched for binary " + bid)
                    continue

            i = b.filter_inputs(i)
            if len(i) == 0:
                if not self.inp_filtered:
                    log.error("Filtering discarded all inputs for binary " + bid)
                    return False
                else:
                    log.warning("Filtering discarded all inputs for binary " + bid)
                    continue
            
            self.bin_inputs[bid] = i

        return True

    # NOTE: I (Loc) added config in so I could pass it in
    def get_run_specs(self, config=None):
        """Returns a list of all of the run specifications for binaries in
        this loader (one run spec for each input it is associated with).
        """
        out = []
        for bid, b in self.binaries.iteritems():
            if bid in self.bin_inputs:
                for inp in self.bin_inputs[bid]:
                    testList = b.get_run_spec(inp, config)
                    for k in testList:
                      out.append(k)
                    #out.append((i) for i in b.get_run_spec(inp))
            else:
                assert self.inp_filtered, bid
                    
        return out

if __name__ == "__main__":
    import sys
    x = load_binary_specs(sys.argv[1])
    for bmk in x:
        print bmk.get_id()
        bmk.props.dump()
