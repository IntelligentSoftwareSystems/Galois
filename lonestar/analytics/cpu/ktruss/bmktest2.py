import bmk2
from bmkprops import graph_bmk, PERF_RE, get_ktruss_checker
import os

class KtrussGaloisBase(graph_bmk):
    bmk = "ktruss"
    algo = None

    def filter_inputs(self, inputs):
        def finput(x):
            if not "symmetric" in x.props.flags: return False
            if x.props.format == 'bin/galois': return True

            return False

        return filter(finput, inputs)

    def get_run_spec(self, bmkinput):
        x = bmk2.RunSpec(self, bmkinput)

        k, ec = get_ktruss_checker(bmkinput, self.config['k'])
        t = int(self.config['t'])

        x.set_binary(self.props._cwd, 'k-truss')
        x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
        assert self.algo is not None
        x.set_arg('-algo=%s' % (self.algo,), bmk2.AT_OPAQUE)
        x.set_arg('-trussNum=%d' % (k,), bmk2.AT_OPAQUE)
        x.set_arg("-t=%d" % (t,), bmk2.AT_OPAQUE)
        x.set_arg('-o=@output', bmk2.AT_TEMPORARY_OUTPUT)
        x.set_checker(bmk2.ExternalChecker(ec))

        x.set_perf(bmk2.PerfRE(r"^\(NULL\),.*, Time,0,0,(?P<time_ms>[0-9]+)$"))
        return x

class KtrussGaloisBSP(KtrussGaloisBase):
    variant = "galois+bsp"
    algo = "bsp"

class KtrussGaloisBSPIm(KtrussGaloisBase):
    variant = "galois+bspIm"
    algo = "bspIm"

class KtrussGaloisBSPCoreThenTruss(KtrussGaloisBase):
    variant = "galois+bspCoreThenTruss"
    algo = "bspCoreThenTruss"

class KtrussGaloisAsync(KtrussGaloisBase):
    variant = "galois+async"
    algo = "async"

        
BINARIES = [KtrussGaloisBSP(),
            KtrussGaloisBSPIm(),
            KtrussGaloisBSPCoreThenTruss(),
            KtrussGaloisAsync(),]
