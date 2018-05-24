import bmk2
from bmkprops import GraphBMKSharedMem
import os

class SharedMemApp(GraphBMKSharedMem):
    """Base class that has default run spec construction behavior for
    most if not all shared memory apps.
    """
    # thread to start from
    startThread = 0
    # thread to end at (inclusive)
    endThread = 56
    # step to use for looping through threads
    step = 7

    def filter_inputs(self, inputs):
        """Ignore inputs that aren't currently supported."""
        def finput(x):
            if x.props.format == 'bin/galois': return True
            if x.props.format == 'mesh': return True
            if x.props.format == 'mesh/nodes': return True
            if x.props.format == 'triangles': return True
            if x.props.format == 'text': return True
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
            else: # mesh
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
            s.set_arg("-n=500000")
            s.set_arg("-steps=1")
            s.set_arg("-seed=0")
        
        return specs

class BCAsync(SharedMemApp):
    relativeAppPath = "betweennesscentrality/bc-async"
    benchmark = "bc-async"

    def get_run_spec(self, bmkinput, config):
        """BC async command line setup"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            # do 5 nodes with edges
            s.set_arg("-numOfOutSources=5") 
        
        return specs

class BCOuter(SharedMemApp):
    relativeAppPath = "betweennesscentrality/betweennesscentrality-outer"
    benchmark = "bc-outer"

class BFS(SharedMemApp):
    relativeAppPath = "bfs/bfs"
    benchmark = "bfs"

class Boruvka(SharedMemApp):
    relativeAppPath = "boruvka/boruvka"
    benchmark = "boruvka"

class BoruvkaMerge(SharedMemApp):
    relativeAppPath = "boruvka/boruvka-merge"
    benchmark = "boruvka-merge"

class Clustering(SharedMemApp):
    relativeAppPath = "clustering/clustering"
    benchmark = "clustering"

    def get_run_spec(self, bmkinput, config):
        """Clustering command line setup"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-numPoints=10000") # num points to cluster
        
        return specs

class ConnectedComponents(SharedMemApp):
    relativeAppPath = "connectedcomponents/connectedcomponents"
    benchmark = "connectedcomponents"

class DelaunayTriangulation(SharedMemApp):
    relativeAppPath = "delaunaytriangulation/delaunaytriangulation"
    benchmark = "delaunaytriangulation"

class DelaunayTriangulationDet(SharedMemApp):
    relativeAppPath = "delaunaytriangulation/delaunaytriangulation-det"
    benchmark = "delaunaytriangulation-det"

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

    #def get_run_spec(self, bmkinput, config):
    #    """Adds matrix completion type"""
    #    specs = self.get_default_run_specs(bmkinput, config)

    #    ## TODO change this to some canonical version
    #    #for s in specs:
    #    #    s.set_arg("-algo=block") # algo type
    #    
    #    return specs


class MCM(SharedMemApp):
    relativeAppPath = "matching/bipartite-mcm"
    benchmark = "mcm"

    def get_run_spec(self, bmkinput, config):
        """Adds bipartite matching specific arguments"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-abmpAlgo")
            s.set_arg("-inputType=generated")
            s.set_arg("-n=1000000") # nodes in each bipartite set
            s.set_arg("-numEdges=100000000") 
            s.set_arg("-numGroups=10000") 
            s.set_arg("-seed=0") # seed for rng; keep it consistent
        
        return specs

class PageRankPullResidual(SharedMemApp):
    relativeAppPath = "pagerank/pagerank-pull-residual"
    benchmark = "pagerank-pull-residual"

    def get_run_spec(self, bmkinput, config):
        """Adds tolerance argument"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-tolerance=0.001") # pagerank tolerance
        
        return specs

class PageRankPullSb(SharedMemApp):
    relativeAppPath = "pagerank/pagerank-pull-sb"
    benchmark = "pagerank-pull-sb"

    def get_run_spec(self, bmkinput, config):
        """Adds tolerance argument"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-tolerance=0.001") # pagerank tolerance
        
        return specs

class PageRankPull(SharedMemApp):
    relativeAppPath = "pagerank/pagerank-pull"
    benchmark = "pagerank-pull"

    def get_run_spec(self, bmkinput, config):
        """Adds tolerance argument"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-tolerance=0.001") # pagerank tolerance
        
        return specs

class PageRankPush(SharedMemApp):
    relativeAppPath = "pagerank/pagerank-push"
    benchmark = "pagerank-push"

    def get_run_spec(self, bmkinput, config):
        """Adds tolerance argument"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-tolerance=0.001") # pagerank tolerance
        
        return specs

# for galois 2.2 version of pagerank
class PageRank2Point2(SharedMemApp):
    relativeAppPath = "pagerank/pagerank"
    benchmark = "pagerank22"

    def get_run_spec(self, bmkinput, config):
        """Adds transpose graph."""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-graphTranspose=%s" % bmkinput.props.ptranspose) 
            s.set_arg("-maxIterations=1000")
        
        return specs

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
    relativeAppPath = "pointstoanalysis/pta"
    benchmark = "pta"

class SpanningTree(SharedMemApp):
    relativeAppPath = "spanningtree/spanningtree"  
    benchmark = "spanningtree"

class SSSP(SharedMemApp):
    relativeAppPath = "sssp/sssp"
    benchmark = "sssp"

    def get_run_spec(self, bmkinput, config):
        """Adds delta argument to runs."""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-delta=8")
        
        return specs

class SurveyPropagation(SharedMemApp):
    relativeAppPath = "surveypropagation/surveypropagation"
    benchmark = "surveypropagation"

    def get_run_spec(self, bmkinput, config):
        """Adds survey prop arguments to runs."""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("9") # random generator seed
            s.set_arg("2000000") # number of vars
            s.set_arg("6000000") # number of clauses
            s.set_arg("4") # vars per clause

            # below are args used by runs on galois website
            #s.set_arg("1000000") # number of vars
            #s.set_arg("3000000") # number of clauses
            #s.set_arg("3") # vars per clause
       
        return specs

class TrianglesNode(SharedMemApp):
    relativeAppPath = "triangles/triangles"
    benchmark = "triangles-node"

    def get_run_spec(self, bmkinput, config):
        """Specifies node version"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-algo=nodeiterator")
        
        return specs

class TrianglesEdge(SharedMemApp):
    relativeAppPath = "triangles/triangles"
    benchmark = "triangles-edge"

    def get_run_spec(self, bmkinput, config):
        """Specifies edge version"""
        specs = self.get_default_run_specs(bmkinput, config)

        for s in specs:
            s.set_arg("-algo=edgeiterator")
        
        return specs

# specification of binaries to run
# apps present in Galois 2.2
#BINARIES = [BarnesHut(), BFS(), BCOuter(), Boruvka(), BoruvkaMerge(), 
#            Clustering(), ConnectedComponents(), DelaunayTriangulation(), DMR(), 
#            GMetis(), IndependentSet(), MCM(), PageRankPull(), PageRankPush(), 
#            PreflowPush(), SpanningTree(), SSSP(), SurveyPropagation()]

# apr 12 run
BINARIES = [BarnesHut(), BFS(), BCAsync(), BCOuter(), Boruvka(), 
            ConnectedComponents(), DelaunayTriangulation(), DMR(), 
            GMetis(), IndependentSet(), MCM(), PageRankPull(), PageRankPush(), 
            PreflowPush(), PointsToAnalysis(), SSSP(), SurveyPropagation(), 
            TrianglesNode(), TrianglesEdge()] 
