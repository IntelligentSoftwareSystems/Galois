import bmk2
from bmkprops import GraphBMKDistApp
import os

################################################################################
# DistApp base class
################################################################################

class DistApp(GraphBMKDistApp):
  """Base class that has default run spec construction behavior for most
  dist apps."""
  # thread to start from
  startThread = 40 
  # thread to end at (inclusive)
  endThread = 40
  # step to use for looping through threads
  step = 10

  # list of hosts to loop through
  testHosts = [1]
  #testHosts = [1, 2, 3]

  # list of cuts to test
  # TODO use hybrid cuts?
  cutsToTest = ["oec", "iec", "cvc"]

  def filter_inputs(self, inputs):
    """Ignore inputs that aren't currently supported; dist apps only 
    support the Galois binary graph format."""
    def finput(x):
      if x.props.format == 'bin/galois': return True
      return False

    return filter(finput, inputs)

  def get_default_run_specs(self, bmkinput, config):
    """Creates default run specifications with common arguments for all
    dist apps and returns them. They can be modified
    later according to the benchmark that you want to run.
    """
    assert config != None # config should be passed through test2.py

    listOfRunSpecs = []

    # TODO add cuts in as well....
    for numThreads in range(self.startThread, self.endThread + 1, self.step):
      if numThreads == 0 and self.step != 1:
        numThreads = 1
      elif numThreads == 0:
        continue

      for numHosts in self.testHosts:
        # TODO no cut if 1 host
        for currentCut in self.cutsToTest:
          # TODO figure out how to get mpirun hooked up to this

          x = bmk2.RunSpec(self, bmkinput)

          # mpirun setup
          x.set_binary("", os.path.expandvars(
                             os.path.join(config.get_var("pathToMPIRun"))))
          x.set_arg("-n=%d" % numHosts)
          # TODO set this in config instead?
          x.set_arg("-hosts=peltier,gilbert,oersted")

          # app setup
          x.set_arg(os.path.expandvars(
                         os.path.join(config.get_var("pathToApps"),
                                      self.relativeAppPath)))
          x.set_arg("-t=%d" % numThreads)

          # set transpose or symm graph flag
          if not (bmkinput.props.file).endswith(".sgr"):
            x.set_arg("-graphTranspose=%s" % bmkinput.props.transpose)
          else:
            x.set_arg("-symmetricGraph")

          nameToAppend = bmkinput.name

          # partition setup
          if numHosts != 1:
            x.set_arg("-partition=%s" % currentCut)
          else:
            currentCut = "single"

          x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
          x.set_arg("-statFile=" +
                    os.path.expandvars(
                      os.path.join(config.get_var("logOutputDirectory"),
                                   self.getUniqueStatFile(numThreads, numHosts,
                                                          currentCut,
                                                          nameToAppend))))

          listOfRunSpecs.append(x)

          # null checkers/perf checkers
          x.set_checker(bmk2.PassChecker())
          x.set_perf(bmk2.ZeroPerf())

          # escape partition loop if only in a single host
          if (currentCut == "single"):
            break

    return listOfRunSpecs

  def get_run_spec(self, bmkinput, config):
    return self.get_default_run_specs(bmkinput, config)


################################################################################
# List of apps to test
################################################################################

class BFSPush(DistApp):
  relativeAppPath = "bfs_push"
  benchmark = "bfs_push"

  def get_run_spec(self, bmkinput, config):
    """Adds source of bfs"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-srcNodeId=%s" % bmkinput.props.source)
      
    return specs

class BFSPull(DistApp):
  relativeAppPath = "bfs_pull"
  benchmark = "bfs_pull"

  def get_run_spec(self, bmkinput, config):
    """Adds source of bfs"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-srcNodeId=%s" % bmkinput.props.source)
      
    return specs

class CCPush(DistApp):
  relativeAppPath = "cc_push"
  benchmark = "cc_push"
  
class CCPull(DistApp):
  relativeAppPath = "cc_pull"
  benchmark = "cc_pull"

class KCorePush(DistApp):
  relativeAppPath = "kcore_push"
  benchmark = "kcore_push"

  def get_run_spec(self, bmkinput, config):
    """Adds kcore num"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-kcore=100")
      
    return specs

class KCorePull(DistApp):
  relativeAppPath = "kcore_pull"
  benchmark = "kcore_pull"

  def get_run_spec(self, bmkinput, config):
    """Adds kcore num"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-kcore=100")
      
    return specs

class PageRankPush(DistApp):
  relativeAppPath = "pagerank_push"
  benchmark = "pagerank_push"
  # TODO max iterations?

class PageRankPull(DistApp):
  relativeAppPath = "pagerank_pull"
  benchmark = "pagerank_pull"
  # TODO max iterations?

class SSSPPush(DistApp):
  relativeAppPath = "sssp_push"
  benchmark = "sssp_push"

  def get_run_spec(self, bmkinput, config):
    """Adds source of sssp"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-srcNodeId=%s" % bmkinput.props.source)
      
    return specs

class SSSPPull(DistApp):
  relativeAppPath = "sssp_pull"
  benchmark = "sssp_pull"

  def get_run_spec(self, bmkinput, config):
    """Adds source of sssp"""
    specs = self.get_default_run_specs(bmkinput, config)

    for s in specs:
      s.set_arg("-srcNodeId=%s" % bmkinput.props.source)
      
    return specs


################################################################################
# Specification of binaries to run
################################################################################

#BINARIES = [BFSPush(), BFSPull()]
BINARIES = [BFSPush(), BFSPull(), CCPush(), CCPull(), KCorePush(), KCorePull(),
            PageRankPush(), PageRankPull(), SSSPPush(), SSSPPull()]
