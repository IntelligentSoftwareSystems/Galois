import bmk2
from bmkprops import GraphBMKSharedMem
import os

class SharedMemApp(GraphBMKSharedMem):
    def filter_inputs(self, inputs):
        """Ignore inputs that aren't currently supported."""
        def finput(x):
            if x.props.format == 'bin/galois': return True
            if x.props.format == 'mesh': return True
            if x.props.format == 'nothing': return True

            return False

        return filter(finput, inputs)

    def get_default_run_specs(self, bmkinput, config):
        """Creates default run specifications with common arguments for all
        shared memory benchmarks and returns them. They can be modified
        later according to the benchmark that you want to run.
        """
        assert config != None
        listOfRunSpecs = []

        for numThreads in range(self.startThread, self.endThread + 1, self.step):
            if numThreads == 0 and self.step != 1:
              numThreads = 1
            elif numThreads == 0:
              continue

            x = bmk2.RunSpec(self, bmkinput)

            x.set_binary("", os.path.join(config.get_var("pathToApps"),
                                          self.relativeAppPath))
            x.set_arg("-t=%d" % numThreads)
            x.set_arg("-statFile=" +
                      os.path.join(config.get_var("logOutputDirectory"),
                                   self.getUniqueStatFile(numThreads, 
                                                          bmkinput.name)))
            if bmkinput.props.format == "nothing":
                pass
            elif bmkinput.props.format != "mesh":
                x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
            else:
                # don't specify with input file flag as it doesn't exist (mesh
                # loads multiple files, so the file specified in the inputdb
                # isn't an actual file
                x.set_arg(bmkinput.props.file)

            listOfRunSpecs.append(x)

            x.set_checker(bmk2.PassChecker())
            x.set_perf(bmk2.ZeroPerf())

        return listOfRunSpecs

    def get_run_spec(self, bmkinput, config):
        return self.get_default_run_specs(bmkinput, config)

class BFS(SharedMemApp):
    relativeAppPath = "bfs/bfs"
    benchmark = "bfs"
    startThread = 80
    endThread = 80
    step = 10

class SSSP(SharedMemApp):
    relativeAppPath = "sssp/sssp"
    benchmark = "sssp"
    startThread = 80
    endThread = 80
    step = 10

    def get_run_spec(self, bmkinput, config):
        """Adds delta argument to runs."""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-delta=8")
        
        return specs

class DMR(SharedMemApp):
    relativeAppPath = "delaunayrefinement/delaunayrefinement"
    benchmark = "dmr"
    startThread = 80
    endThread = 80
    step = 10

class BarnesHut(SharedMemApp):
    relativeAppPath = "barneshut/barneshut"
    benchmark = "barneshut"
    startThread = 80
    endThread = 80
    step = 10

    def get_run_spec(self, bmkinput, config):
        """Adds barnes hut specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-n=50000")
            s.set_arg("-steps=1")
            s.set_arg("-seed=0")
        
        return specs

#BINARIES = [BFS(), SSSP(), DMR()]
# specification of binaries to run
BINARIES = [BarnesHut(),]
