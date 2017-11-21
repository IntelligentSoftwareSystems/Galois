#
# config.py
#
# bmk2.cfg reader for bmk2.
#
# Copyright (c) 2015, 2016 The University of Texas at Austin
#
# Author: Sreepathi Pai <sreepai@ices.utexas.edu>
#
# Intended to be licensed under GPL3

import glob
import os
import logging
import ConfigParser

log = logging.getLogger(__name__)

# essentially enums for globs
FT_BISPEC = 1
FT_INPUTDB = 2
FT_INPUTPROPS = 3
FT_INPUTPROC = 4

FT_FIRST = FT_BISPEC
FT_LAST = FT_INPUTPROC

# file types you can have multiple of
FT_MULTIPLE_OKAY = set()
# file types that aren't necessary
FT_ZERO_OKAY = set([FT_INPUTPROPS, FT_INPUTPROC])
# file types that are possible to load + their expected file extensions
FT_GLOBS = {FT_BISPEC: '*.bispec', 
            FT_INPUTDB: '*.inputdb',
            FT_INPUTPROPS: '*.inputprops', 
            FT_INPUTPROC: None}

class Config(object):
    """Class that provides an interface to a BMK2 configuration file"""
    def __init__(self, metadir, inpproc = None):
        """Setup data structures and load the config file into the object

        Keyword Arguments:
        metadir -- the directory with the configuration information
        inpproc -- TODO figure out what this is
        """
        self.metadir = metadir
        self.okay = False
        self.files = {}
        self.disable_binaries = set()
        self.site = None

        if inpproc is not None:
            self.files = {FT_INPUTPROC: inpproc}

        self._load_config()

    def set_file(self, f, ty, multiple = False):
        """Save a file name into a particular class of files.

        Keyword Arguments:
        f -- the file name
        ty -- the type of the passed in file (see FT_GLOBS for file types)
        multiple -- specifies if you can have multiple files of some type
        """
        assert not (ty < FT_FIRST or ty > FT_LAST), "Invalid file type: %s" % (ty,)
        
        if multiple:
            assert ty in FT_MULTIPLE_OKAY, "File type %d not in multiple" % (ty,)

        if not (os.path.exists(f) and os.path.isfile(f)):
            log.error("File '%s' (file type: %d) does not exist or is not a file" % 
                      (f, ty))
            return False

        if ty not in self.files:
            if multiple:
                self.files[ty] = [f]
            else:
                self.files[ty] = f
        else:
            if multiple:
                self.files[ty].append(f)
            else:
                log.warning("Overwriting file type %d (currently: %s) with %s" % 
                            (ty, self.files[ty], f))
                self.files[ty] = f

        return True

    def get_file(self, ty):
        """Get the file of a particular type from this object."""
        if ty not in self.files:
            return None

        return self.files[ty]
        
    def _site_specific_cfg(self, x):
        """TODO"""
        sitefiles = glob.glob(os.path.join(self.metadir, "SITE-IS.*"))

        if len(sitefiles) > 1:
            log.error("Only one sitefile should exist. Currently, multiple sitefiles exist: '%s'" % (sitefiles,))
        elif len(sitefiles) == 0:
            log.info("No sitefile found.")
        else:
            p = sitefiles[0].rindex(".")
            self.site = sitefiles[0][p+1:]
            log.info("Site set to '%s'." % (self.site,))
            sscfg = os.path.join(self.metadir, "bmk2.cfg." + self.site)

            if not os.path.exists(sscfg):
                log.warning("No site-specific configuration '%s' found." % (sscfg,))
            else:
                log.info("Loading site-specific configuration from '%s'." % (sscfg,))
                y = self._read_config(sscfg,)
                
                for s in y.sections():
                    for n, v in y.items(s):
                        if not self.cfg.has_section(s):
                            self.cfg.add_section(s)

                        log.info("Setting site-specific [%s]:%s to '%s'" % (s, n, v))
                        self.cfg.set(s, n, v)                

                return True

        return False

    def _read_config(self, f):
        """TODO"""
        x = ConfigParser.SafeConfigParser()

        with open(f, "rb") as fp:
            x.readfp(fp)

            try:
                version = x.get("bmk2", "version")
                if version != "2":
                    log.error("%s: Unknown config version %s" % (self.config_file, version,))
                    return False
            except ConfigParser.NoOptionError:
                log.error("%s: Unable to read version" % (self.config_file,))
                return False

            return x

    def _load_config(self):
        """Open a configuration file: it must have a version specification for this
        function to exit cleanly. Saves the opened configuration file to cfg.
        """
        self.cfg = None

        if not (os.path.exists(self.metadir) and os.path.isdir(self.metadir)):
            log.error("Metadir '%s' does not exist or is not a directory" % (self.metadir,))
            return False

        self.config_file = os.path.join(self.metadir, "bmk2.cfg")
        if not (os.path.exists(self.config_file) and os.path.isfile(self.config_file)):
            log.error("Configuration file '%s' does not exist" % (self.config_file,))
            return False

        x = self._read_config(self.config_file)
        if x == False:
            return x

        self.cfg = x
        if self._site_specific_cfg(x) == False:
            return False
        #x = ConfigParser.SafeConfigParser()
        #with open(self.config_file, "rb") as f:
        #    x.readfp(f)

        #    # only reads version 2 bmk2 config files
        #    try:
        #        version = x.get("bmk2", "version")
        #        if version != "2":
        #            log.error("%s: Unknown config version %s" % 
        #                      (self.config_file, version,))
        #            return False
        #    except ConfigParser.NoOptionError:
        #        log.error("%s: Unable to read version" % (self.config_file,))
        #        return False

        #    self.cfg = x

    def load_config(self):
        """Load the file types and disabled binaries as specified by the 
        loaded configuration file.
        """
        x = self.cfg
        if not x:
            return False

        # load each file type, save names
        for prop, ty in [("inpproc", FT_INPUTPROC),
                         ("inputdb", FT_INPUTDB),
                         ("inputprops", FT_INPUTPROPS),
                         ("bispec", FT_BISPEC)]:
            try:
                # section bmk2 with a particular property
                val = x.get("bmk2", prop)
                val = os.path.join(self.metadir, val)
                if self.set_file(val, ty):
                    log.info("%s: Loaded file type %d ('%s')" % 
                             (self.config_file, ty, val))
                else:
                    return False
            except ConfigParser.NoOptionError:
                log.debug("%s: File type %d (property: '%s') not specified" % 
                          (self.config_file, ty, prop))

        # save disabled binaries into object
        try:
            val = x.get("bmk2", "disable_binaries")
            self.disable_binaries = set([xx.strip() for xx in val.split(",")])
        except ConfigParser.NoOptionError:
            pass

        # TODO find out what this is
        self.bin_config = None
        if x.has_section("default-config"):
            self.bin_config = self.section_to_dict(x, "default-config")

        self.cfg = x
        return True

    def section_to_dict(self, cfgobj, section):        
        """TODO"""
        kv = cfgobj.items(section)

        o = set()
        for kk in kv:
            if kk[0] in o:
                log.warning("Duplicated key '%s' in section '%s'", kk[0], section)

            o.add(kk[0])

        return dict(kv)

    def load_bin_config(self, config_sections):
        """TODO"""
        x = self.cfg
        if not x:
            return False
        
        ok = True
        out = []
        for s in config_sections:
            if not x.has_section(s):
                log.error("Configuration section '%s' not found" % (s,))
                ok = False
            else:
                out.append(self.section_to_dict(x, s))

        if ok:
            nout = {}
            for o in out:
                if 'type' in o and o['type'] == 'bmk2config':
                    if 'disable_binaries' in o:
                        v = set([xx.strip() for xx in o['disable_binaries'].split(",")])
                        self.disable_binaries = self.disable_binaries.union(v)
                    else:
                        # TODO: handle other configuration specific things?
                        pass
                else:
                    nout.update(o)

            if len(nout):
                if self.bin_config is None:
                    self.bin_config = {}

                self.bin_config.update(nout)

            return True

        return ok
        

    def load_var_config(self, varconfigs):
        """TODO"""
        o = {}
        for vv in varconfigs:
            va, vl = vv.split("=")
            o[va] = vl

        if self.bin_config is None:
            self.bin_config = {}

        # TODO: warn of command line config over-riding?
        self.bin_config.update(o)
        return True


                    
    def get_var(self, key, default = None, sec = "bmk2"):
        """Loads a variable from a particular secion of the loaded
        config file

        Keyword Arguments:
        key -- variable to load
        default -- default value to return
        sec -- section in the config file to search
        """
        try:
            return self.cfg.get(sec, key)
        except ConfigParser.NoOptionError:
            return default

    def auto_set_files(self):
        """Load any required remaining files in the metadata directory that have
        not been loaded into the files structure of the object yet.
        """
        for ty in range(FT_FIRST, FT_LAST):
            if ty not in self.files and FT_GLOBS[ty] is not None:
                matches = glob.glob(os.path.join(self.metadir, FT_GLOBS[ty]))

                if len(matches) == 0:
                    if ty not in FT_ZERO_OKAY:
                        log.error("File type %d (%s) required, but not found in %s" % 
                                  (ty, FT_GLOBS[ty], self.metadir))
                        return False
                elif len(matches) == 1:
                    log.info("File type %d auto set to %s" % (ty, matches[0]))
                    if not self.set_file(matches[0], ty, False):
                        return False
                elif len(matches) > 1:
                    if ty not in FT_MULTIPLE_OKAY:
                        log.error("Multiple matches found for file type %d (%s) in %s, must specify only one." % 
                                  (ty, FT_GLOBS[ty], self.metadir))
                        return False
                    else:
                        for f in matches:
                            if not self.set_file(f, ty, True):
                                return False

        return True

__all__  = ['FT_BISPEC', 'FT_INPUTDB', 'FT_INPUTPROC', 'FT_INPUTPROPS',
            'Config']

if __name__ == "__main__":
    """Load configuration files specified by the configuration data.
    The directory to the configuration data should be the second argument,
    and there should be a bmk2.cfg file in the directory.
    """
    import sys
    logging.basicConfig(level=logging.DEBUG)
    inpproc = None

    if len(sys.argv) > 2:
        inpproc = sys.argv[2]

    x = Config(sys.argv[1], inpproc)
    if x.load_config():
        if x.auto_set_files():
            print "LOADED CONFIG"
