# cython: cdivision = True

from galois.shmem cimport *
from cython.operator cimport preincrement, dereference as deref

ctypedef atomic[uint32_t] atomuint32_t
ctypedef atomic[uint64_t] atomuint64_t
##############################################################################
## Pagerank implementation
###############################################################################
#
# Struct for Pagerank
#
cdef struct NodeTy:
    float rank
    uint32_t nout

ctypedef LC_CSR_Graph[NodeTy, void, dummy_true] Graph

# Cython bug: using a nested class from a previous typedef doesn't
# work for the time being. Instead, the full template specialization
# must be used to get the member type.
ctypedef LC_CSR_Graph[NodeTy, void, dummy_true].GraphNode GNode

#
# Constants for Pagerank
#
cdef float ALPHA = 0.85
cdef float INIT_RESIDUAL = 1 - ALPHA;
cdef float TOLERANCE   = 1.0e-3;
cdef uint32_t MAX_ITER = 1000;

#
# Initialization for Pagerank
#
cdef void InitializePR(Graph *g):
    cdef unsigned long numNodes = g[0].size()
    cdef NodeTy *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        #gPrint(n,"\n")
        data = &g[0].getData(n)
        data[0].rank = INIT_RESIDUAL
        data[0].nout = 0

cdef void printValuePR(Graph *g):
    cdef unsigned long numNodes = g[0].size()
    cdef NodeTy *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        #gPrint(n,"\n")
        data = &g[0].getData(n)
        #if(data[0].nout.load() > 0):
        gPrint(data[0].rank, b"\n")

#
# Operator for computing outdegree of nodes in the Graph
#
cdef void computeOutDeg_operator(Graph *g, LargeArray[atomuint64_t] *largeArray, GNode n) nogil:
    cdef: 
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ei
        GNode dst
        #NodeTy *dst_data
        
    ii = g[0].edge_begin(n)
    ei = g[0].edge_end(n)
    while ii != ei:
            dst = g[0].getEdgeDst(ii)
            largeArray[0][<size_t>dst].fetch_add(1)
            preincrement(ii)
    
#
# Operator for assigning outdegree of nodes in the Graph
#
cdef void assignOutDeg_operator(Graph *g, LargeArray[atomuint64_t] *largeArray, GNode n) nogil:
    cdef NodeTy *src_data
        
    src_data = &g[0].getData(n)
    src_data.nout = largeArray[0][<size_t>n].load()
#
#
# Main callsite for computing outdegree of nodes in the Graph
#
cdef void computeOutDeg(Graph *graph):
    cdef: 
        uint64_t numNodes = graph[0].size()
        LargeArray[atomuint64_t] largeArray

    largeArray.allocateInterleaved(numNodes)
    with nogil:
        do_all(iterate(graph[0].begin(), graph[0].end()),
                        bind_leading(&computeOutDeg_operator, graph, &largeArray), steal(),
                        loopname("ComputeDegree"))

        do_all(iterate(graph[0].begin(), graph[0].end()),
                        bind_leading(&assignOutDeg_operator, graph, &largeArray))


#
# Operator for PageRank
#
cdef void pagerankPullTopo_operator(Graph *g, GReduceMax[float] *max_delta, GNode n) nogil:
    cdef: 
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ei
        GNode dst
        NodeTy *dst_data
        NodeTy *src_data
        float sum = 0
        float value = 0
        float diff = 0;
    ii = g[0].edge_begin(n, FLAG_UNPROTECTED)
    ei = g[0].edge_end(n, FLAG_UNPROTECTED)
    src_data = &g[0].getData(n)
    while ii != ei:
            dst_data = &g[0].getData(g[0].getEdgeDst(ii), FLAG_UNPROTECTED)
            sum += dst_data[0].rank / dst_data[0].nout
            preincrement(ii)
    value = sum * ALPHA + (1.0 - ALPHA)
    diff = fabs(value - src_data[0].rank);
    src_data[0].rank = value
    max_delta[0].update(diff)

#
# Pagerank routine: Loop till convergence
#
cdef void pagerankPullTopo(Graph *graph, uint32_t max_iterations) nogil:
    cdef: 
        GReduceMax[float] max_delta
        float delta = 0
        uint32_t iteration = 0
        Timer T

    T.start()
    while(1):
        with nogil:
            do_all(iterate(graph[0].begin(), graph[0].end()),
                        bind_leading(&pagerankPullTopo_operator, graph, &max_delta), steal(),
                        loopname("PageRank"))

        delta = max_delta.reduce()
        iteration += 1
        if(delta <= TOLERANCE or iteration >= max_iterations):
            break
        max_delta.reset();
    
    T.stop()
    gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")
    if(iteration >= max_iterations):
        gPrint(b"ERROR : failed to converge in ", iteration, b" iterations\n")
    

#
# Main callsite for Pagerank
#   
def pagerank(int numThreads, uint32_t max_iterations, string filename):
    cdef int new_numThreads = setActiveThreads(numThreads)
    gPrint(b"Running Pagerank on : ", filename, b"\n")
    if new_numThreads != numThreads:
        print("Warning, using fewer threads than requested")
    
    print("Using {0} thread(s).".format(new_numThreads))
    cdef Graph graph
    
    ## Read the CSR format of graph
    ## directly from disk.
    graph.readGraphFromGRFile(filename)
    
    InitializePR(&graph)
    computeOutDeg(&graph)
    #printValuePR(&graph)
    pagerankPullTopo(&graph, max_iterations)
    #printValuePR(&graph)
    
   
