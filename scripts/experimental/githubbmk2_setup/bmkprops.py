import bmk2
import datetime

TIME_FMT = "%Y-%m-%d %H:%M:%S"

class GraphBMKSharedMem(bmk2.Binary):
    """Base class for shared memory benchmarks to inherit from. Subclasses
    need to specify benchmark name + number of threads.
    """
    def __init__(self):
        """Initialize shared mem properties."""
        self.props = GraphBMKSharedMemProps(self.benchmark)
        
    def get_id(self):
        """Return the id of this benchmark."""
        return "%s" % (self.benchmark)

    def getUniqueStatFile(self, numThreads, graphName):
        """Get a statfile name given num threads + graph name being used."""
        timeNow = datetime.datetime.now().strftime(TIME_FMT).replace(" ", "_")

        return ("%s_%d_%s_%s.log" % (self.benchmark, numThreads, graphName,
                               timeNow))

class GraphBMKSharedMemProps(bmk2.Properties):
    """Properties pertaining to shared memory."""
    def __init__(self, benchmark):
        self.benchmark = benchmark
