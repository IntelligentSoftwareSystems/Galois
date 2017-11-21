#
# opdb.py
#
# Object properties database for bmk2.  Sections in CFG files indicate
# objects, section keys indicate properties.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import ConfigParser
from collections import OrderedDict
import os
import glob
import sys

def cfg_get(fn, section, key, default=None):
    """Wrapper for ConfigParser functions that allows you to return a default
    value on failure.

    Keyword Arguments:
    fn -- config parser function to use
    section -- section of config file to read
    key -- key to attempt to access from section
    default -- default value to return on error
    """
    try:
        v = fn(section, key)
        return v
    except ConfigParser.NoOptionError:
        return default

class ObjectProps(object):
    pass

class ObjectPropsCFG(ObjectProps):
    """Read/Write a .cfg file as a object property file.
    
       Sections names indicate objects, section keys indicate properties."""
    def __init__(self, filename, fmt, acceptable_versions):
        """Initializer.

        Keyword Arguments:
        filename -- config file to read
        fmt -- format of the config file (should be section name in config
        file that has meta information); 
        acceptable_versions -- versions of the config file that are allowed
        """
        self.filename = filename
        self.fmt = fmt
        self.acceptable_versions = acceptable_versions
        self.meta = None
        self.objects = OrderedDict() # contains section -> vars in section
        self.site = None

    def _site_specific_cfg(self, x):
        d = os.path.dirname(self.filename)
        sitefiles = glob.glob(os.path.join(d, "SITE-IS.*"))

        if len(sitefiles) > 1:
            print >>sys.stderr, ("Only one sitefile should exist. Currently, multiple sitefiles exist: '%s'" % (sitefiles,))
        elif len(sitefiles) == 0:
            print >>sys.stderr, ("No sitefile found.")
        else:
            p = sitefiles[0].rindex(".")
            self.site = sitefiles[0][p+1:]
            print >>sys.stderr, ("Site set to '%s'." % (self.site,))
            sscfg = self.filename + "." + self.site

            if not os.path.exists(sscfg):
                print >>sys.stderr, ("Site-specific input db '%s' not found." % (sscfg,))
            else:
                print >>sys.stderr, ("Loading site-specific '%s'." % (sscfg,))

                y = ConfigParser.SafeConfigParser()

                with open(sscfg, "rb") as f:
                    y.readfp(f)

                    v = cfg_get(y.get, self.fmt, "version")

                    self.version = v

                    if not self.check_version(v):
                        av = [str(v) for v in self.acceptable_versions]
                        if v:
                            print >>sys.stderr, "Unknown version: %s (acceptable: %s)" % (v, ", ".join(av))
                        else:
                            print >>sys.stderr, "Unable to determine version (acceptable: %s)" % (", ".join(av))

                    for s in ("bmktest2", ):
                        for n, v in y.items(s):
                            if not x.has_section(s):
                                x.add_section(s)
                                
                            print >>sys.stderr, ("Setting site-specific [%s]:%s to '%s'" % (s, n, v))
                            x.set(s, n, v)                

                return True

        return False

    def check_version(self, version):
        """Check if a version is allowed by this reader."""
        return version in self.acceptable_versions

    def update_props(self, props):
        return props

    def parse_section(self, cfg, section): 
        """Given a dictionary that represents the parse of some section of
        a config file, return them as an ordered dictionary.

        Keyword Arguments:
        cfg -- dictionary that contains parse results
        section -- section to retrieve
        """
        d = OrderedDict(cfg.items(section))
        d = self.update_props(d)
        return d

    def unparse_section(self, section):
        return section
    
    def post_load(self):
        return True

    def load(self):
        """Load the configuration file and parse its sections."""
        x = ConfigParser.SafeConfigParser()

        out = OrderedDict()
        with open(self.filename, "rb") as f:
            x.readfp(f)

            v = cfg_get(x.get, self.fmt, "version")

            self.version = v

            if not self.check_version(v):
                av = [str(v) for v in self.acceptable_versions]
                if v:
                    print >>sys.stderr, "Unknown version: %s (acceptable: %s)" % (v, ", ".join(av))
                else:
                    print >>sys.stderr, "Unable to determine version (acceptable: %s)" % (", ".join(av))
                
            self._site_specific_cfg(x)

            # save vars in a section to dictionary
            for s in x.sections():
                if s == self.fmt: 
                    self.meta = self.parse_section(x, s)
                else:
                    if s in out:
                        print >>sys.stderr, "Warning: Duplicate section '%s', overwriting" % (s,)

                    out[s] = self.parse_section(x, s)

            self.objects = out
            return self.post_load()

        return False

    def save(self, fn = None):
        """Save the parsed configuration file back into another file."""
        def write_items(cfg, section, items):
            for k, v in items.iteritems():
                cfg.set(section, k, v)

        x = ConfigParser.SafeConfigParser()
        
        assert self.filename or fn, "Both filename and fn cannot be empty."
        if not fn: fn = self.filename

        x.add_section(self.fmt)
        write_items(x, self.fmt, self.unparse_section(self.meta))

        for s in self.objects:
            x.add_section(s)
            write_items(x, s, self.unparse_section(self.objects[s]))

        with open(fn, "wb") as f:
            x.write(f)
        
    def __iter__(self):
        return iter(self.objects.itervalues())
