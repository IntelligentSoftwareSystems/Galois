import bmk2
from bmkprops import GraphBMKSharedMem
import os

class SharedMemApp(GraphBMKSharedMem):
    """Base class that has default run spec construction behavior for
    most if not all shared memory apps.
    """
    # thread to start from
    startThread = 80 
    # thread to end at (inclusive)
    endThread = 80
    # step to use for looping through threads
    step = 10

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
        assert config != None # config should be passed through test2.py
        listOfRunSpecs = []

        for numThreads in range(self.startThread, self.endThread + 1, self.step):
            if numThreads == 0 and self.step != 1:
              numThreads = 1
            elif numThreads == 0:
              continue

            x = bmk2.RunSpec(self, bmkinput)

            x.set_binary("", os.path.expandvars(
                               os.path.join(config.get_var("pathToApps"),
                                          self.relativeAppPath)))
            x.set_arg("-t=%d" % numThreads)

            nameToAppend = bmkinput.name

            if bmkinput.props.format == "nothing":
                nameToAppend = "gen"
                pass
            elif bmkinput.props.format != "mesh":
                x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
            else:
                # don't specify with input file flag as it doesn't exist (mesh
                # loads multiple files, so the file specified in the inputdb
                # isn't an actual file
                x.set_arg(bmkinput.props.file)

            x.set_arg("-statFile=" +
                      os.path.expandvars(
                        os.path.join(config.get_var("logOutputDirectory"),
                                     self.getUniqueStatFile(numThreads, 
                                     nameToAppend))
                      ))

            listOfRunSpecs.append(x)

            x.set_checker(bmk2.PassChecker())
            x.set_perf(bmk2.ZeroPerf())

        return listOfRunSpecs

    def get_run_spec(self, bmkinput, config):
        return self.get_default_run_specs(bmkinput, config)

################################################################################

class BarnesHut(SharedMemApp):
    relativeAppPath = "barneshut/barneshut"
    benchmark = "barneshut"

    def get_run_spec(self, bmkinput, config):
        """Adds barnes hut specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-n=50000")
            s.set_arg("-steps=1")
            s.set_arg("-seed=0")
        
        return specs

class BCInner(SharedMemApp):
    relativeAppPath = "betweennesscentrality/betweennesscentrality-inner"
    benchmark = "bc-inner"

class BCOuter(SharedMemApp):
    relativeAppPath = "betweennesscentrality/betweennesscentrality-outer"
    benchmark = "bc-outer"

class BFS(SharedMemApp):
    relativeAppPath = "bfs/bfs"
    benchmark = "bfs"

class ConnectedComponents(SharedMemApp):
    relativeAppPath = "connectedcomponents/connectedcomponents"
    benchmark = "connectedcomponents"

class DMR(SharedMemApp):
    relativeAppPath = "delaunayrefinement/delaunayrefinement"
    benchmark = "dmr"

class GMetis(SharedMemApp):
    relativeAppPath = "gmetis/gmetis"
    benchmark = "gmetis"

    def get_run_spec(self, bmkinput, config):
        """Adds gmetis specific arguments (num partitions)"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("256") # num of partitions
        
        return specs

class IndependentSet(SharedMemApp):
    relativeAppPath = "independentset/independentset"
    benchmark = "independentset"

class MatrixCompletion(SharedMemApp):
    relativeAppPath = "matrixcompletion/mc"
    benchmark = "matrixcompletion"

class MCM(SharedMemApp):
    relativeAppPath = "matching/bipartite-mcm"
    benchmark = "mcm"

    def get_run_spec(self, bmkinput, config):
        """Adds max card bipartite matching specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-inputType=generated")
            s.set_arg("-n=1000000") # nodes in each bipartite set
            s.set_arg("-numEdges=100000000") 
            s.set_arg("-numGroups=10000") 
            s.set_arg("-seed=0") # seed for rng; keep it consistent
        
        return specs

class PageRank(SharedMemApp):
    relativeAppPath = "pagerank/pagerank"
    benchmark = "pagerank"

    # TODO specify tolerance?

# TODO this is currently crashing
class PreflowPush(SharedMemApp):
    relativeAppPath = "preflowpush/preflowpush"
    benchmark = "preflowpush"

    def get_run_spec(self, bmkinput, config):
        """Adds preflow push specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("0") # source id
            s.set_arg("100") # sink id
        
        return specs

class PointsToAnalysis(SharedMemApp):
    relativeAppPath = "pta/pta"
    benchmark = "pta"

    def get_run_spec(self, bmkinput, config):
        """Adds pta specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("0") # TODO what are contraints?
        
        return specs

class SSSP(SharedMemApp):
    relativeAppPath = "sssp/sssp"
    benchmark = "sssp"

    def get_run_spec(self, bmkinput, config):
        """Adds delta argument to runs."""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-delta=8")
        
        return specs

BINARIES = [BFS()]
#BINARIES = [BFS(), SSSP(), DMR()]
# specification of binaries to run
#BINARIES = [BCInner(), BCOuter()]
