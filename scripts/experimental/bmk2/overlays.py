import core
import logging

log = logging.getLogger(__name__)

class Overlay(object):
    """Class that holds params/env vars to "overlay" into the command line options
    of some run."""
    def __init__(self, env = {}, binary = None, args = []):
        self.env = env
        self.binary = binary
        self.args = args
        self.tmpfiles = {}

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None, logfiles = None):
        """Overlay arguments/env vars into a command line (i.e. add to it)."""
        if env is None:
            new_env = None
        else:
            new_env = env.copy()
            new_env.update(self.env)

        new_cmdline = []
        if self.binary:
            new_cmdline.append(self.binary)

        for a, aty in self.args:
            if aty == core.AT_INPUT_FILE_IMPLICIT:
                continue

            if aty == core.AT_TEMPORARY_OUTPUT:
                th, self.tmpfiles[a] = tempfile.mkstemp(prefix="test-ov-")
                os.close(th)
                log.debug("Created temporary file '%s' for overlay parameter '%s'" % 
                          (self.tmpfiles[a], a))
                a = self.tmpfiles[a]
            elif aty == core.AT_TEMPORARY_INPUT:
                a = inherit_tmpfiles[a]
            elif aty == core.AT_LOG:
                a = logfiles[a]

            new_cmdline.append(a)

        new_cmdline += cmdline

        return new_env, new_cmdline

    def cleanup(self):
        """Cleanup temp files created."""
        for a, f in self.tmpfiles.iteritems():
            os.unlink(f)

    def __str__(self):
        ev = ["%s=%s" % (k, v) for k, v in self.env.iteritems()]
        return "%s %s" % (" ".join(ev), self.cmd_line_c)

class CUDAProfilerOverlay(Overlay):
    def __init__(self, profile_cfg = None, profile_log = None):
        env = {'CUDA_PROFILE': '1'}
        if profile_cfg: env['CUDA_PROFILE_CONFIG'] = profile_cfg
        if profile_log: env['CUDA_PROFILE_LOG'] = profile_log

        self.profile_log = profile_log
        self.collect = logging.getLevelName('COLLECT')
        super(CUDAProfilerOverlay, self).__init__(env)

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        if self.profile_log is not None:
            self.env['CUDA_PROFILE_LOG'] = core.create_log(self.profile_log, run)

        if self.profile_log:
            log.log(self.collect, '{rsid} {runid} cuda/profiler {logfile}'.format(rsid=run.rspec.get_id(), runid=run.runid, logfile=self.env['CUDA_PROFILE_LOG']))
        else:
            log.log(self.collect, '{rsid} {runid} cuda/profiler cuda_profile_0.log'.format(rsid=run.rspec.get_id(), runid=run.runid))
        
        return super(CUDAProfilerOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles)

class NVProfOverlay(Overlay):
    def __init__(self, profile_cfg = None, profile_log = None, profile_db = False, profile_analysis = False, system_profiling = False):
        args = [(x, core.AT_OPAQUE) for x in profile_cfg.strip().split()]

        if profile_db or profile_analysis:
            args += [('-o', core.AT_OPAQUE), ('@nvprofile', core.AT_LOG)]
            if profile_analysis:
                args += [('--analysis-metrics', core.AT_OPAQUE)]
        else:
            args += [(x, core.AT_OPAQUE) for x in "--csv --print-gpu-trace".split()]
            args += [('--log-file', core.AT_OPAQUE), ('@nvprofile', core.AT_LOG)]

        if system_profiling:
            args += [('--system-profiling', core.AT_OPAQUE), ('on', core.AT_OPAQUE)]

        self.profile_cfg = profile_cfg
        self.profile_log = profile_log
        self.profile_db = profile_db or profile_analysis

        self.collect = logging.getLevelName('COLLECT')
        super(NVProfOverlay, self).__init__(binary="nvprof", args=args)

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        if self.profile_log is not None:
            logfile = core.create_log(self.profile_log, run)
        else:
            if self.profile_db:
                logfile = 'cuda_profile_0.nvprof'
            else:
                logfile = 'cuda_profile_0.log'

        log.log(self.collect, '{rsid} {runid} cuda/nvprof {logfile}'.format(rsid=run.rspec.get_id(), runid=run.runid, logfile=logfile))
        
        return super(NVProfOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles, {'@nvprofile': logfile})

class TmpDirOverlay(Overlay):
    def __init__(self, tmpdir):
        env = {'TMPDIR': tmpdir}
        super(TmpDirOverlay, self).__init__(env)

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        return super(TmpDirOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles)

class CLDeviceOverlay(Overlay):
    def __init__(self, cmdline_template, cl_platform, cl_device):
        super(CLDeviceOverlay, self).__init__({})
        self.cmdline_template = cmdline_template
        self.cl_platform = cl_platform
        self.cl_device = cl_device
        self.cmdline = cmdline_template.format(platform = cl_platform, device = cl_device).split(" ")

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        return super(CLDeviceOverlay, self).overlay(run, env, cmdline + self.cmdline, inherit_tmpfiles, {})

class Bmk2RTEnvOverlay(Overlay):
    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        self.env['BMK2'] = "1"
        if isinstance(run.rspec, core.RunSpec):
            self.env['BMK2_BINID'] = run.rspec.bid
            self.env['BMK2_INPUTID'] = run.rspec.input_name

        if run.runid is not None:
            self.env['BMK2_RUNID'] = run.runid

        return super(Bmk2RTEnvOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles)


_instr_overlay_file = {}

class GGCInstrOverlay(Overlay):
    @staticmethod
    def read_map_file(mapfile):
        out = {}
        f = open(mapfile, "r")
        for l in f:
            ls = l.strip().split(' ', 4)
            bmkinput, uniqid, ty, fn, p = ls

            if ty == "ggc/kstate" and bmkinput not in out:
                out[bmkinput] = (uniqid, os.path.dirname(p))

        f.close()

        return out

    def __init__(self, mapfile):        
        if mapfile not in _instr_overlay_file:
            _instr_overlay_file[mapfile] = GGCInstrOverlay.read_map_file(mapfile)
            
        self.mapfile = _instr_overlay_file[mapfile]
        super(GGCInstrOverlay, self).__init__()
    
    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        if run.rspec.get_id() in self.mapfile:
            uid, dirname = self.mapfile[run.rspec.get_id()]

            self.env['INSTR_UNIQID'] = uid
            self.env['INSTR_TRACE_DIR'] = dirname + "/"

        return super(GGCInstrOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles)


class MeasureEnergyOverlay(Overlay):
    def __init__(self):
        super(MeasureEnergyOverlay, self).__init__(binary=os.path.join(os.path.dirname(__file__), "measure_energy.py"))

    def overlay(self, run, env, cmdline, inherit_tmpfiles = None):
        return super(MeasureEnergyOverlay, self).overlay(run, env, cmdline, inherit_tmpfiles)


def add_overlay(rspecs, overlay, *args, **kwargs):
    """Add an overlay to a series of rspecifications."""
    for r in rspecs:
        r.add_overlay(overlay(*args, **kwargs))
