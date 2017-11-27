import bmk2
import datetime

TIME_FMT = "%Y-%m-%d %H:%M:%S"

class GraphBMKDistApp(bmk2.Binary):
  """Base class for dist apps to inherit from. Subclasses specify benchmark
  name + number of threads + number of hosts."""
  def __init__(self):
    """Initialize dist app properties."""
    self.props = GraphBMKDistAppProps(self.benchmark)
        
  def get_id(self):
    """Return the id of this benchmark."""
    return "%s" % (self.benchmark)

  def getUniqueStatFile(self, numThreads, numHosts, currentCut, graphName):
    """Get a statfile name given num threads + graph name being used."""
    timeNow = datetime.datetime.now().strftime(TIME_FMT).replace(" ", "_")

    return ("%s_t=%d_n=%d_%s_%s_%s.log" % (self.benchmark, numThreads, numHosts,
                                           graphName, currentCut, timeNow))

class GraphBMKDistAppProps(bmk2.Properties):
  """Properties pertaining to a dist app."""
  def __init__(self, benchmark):
    self.benchmark = benchmark
