# cython: cdivision= True
from galois.shmem cimport *
from cython.operator cimport preincrement, dereference as deref
from libstd.atomic cimport atomic

ctypedef uint32_t ComponentTy
ctypedef atomic[ComponentTy] AtomicComponentTy
ctypedef atomic[uint32_t] atomuint32_t 

#
# Struct for CC
#
cdef struct NodeTy:
    AtomicComponentTy comp_current
    ComponentTy comp_old

ctypedef LC_CSR_Graph[NodeTy, void, dummy_true] Graph

# Cython bug: using a nested class from a previous typedef doesn't
# work for the time being. Instead, the full template specialization
# must be used to get the member type.
ctypedef LC_CSR_Graph[NodeTy, void, dummy_true].GraphNode GNode


#
# Initialization for Components
#
cdef void initializeCompnents(Graph *g):
    cdef:
        unsigned long numNodes = g[0].size()
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ei
        NodeTy *data
    for n in range(numNodes):
        data = &g[0].getData(n)
        data[0].comp_current.store(n)
        data[0].comp_old = numNodes

##
# LabelProp algorithm operator
##
cdef void labelPropOperator(Graph *g, bool *work_done, GNode n) nogil:
    cdef: 
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[NodeTy, void, dummy_true].edge_iterator ei
        NodeTy *src_data
        NodeTy *dst_data
    src_data = &g[0].getData(n, FLAG_UNPROTECTED)
    if(src_data[0].comp_old > src_data[0].comp_current.load()):
        src_data[0].comp_old = src_data[0].comp_current.load()
        work_done[0] = 1        
        ii = g[0].edge_begin(n, FLAG_UNPROTECTED)
        ei = g[0].edge_end(n, FLAG_UNPROTECTED)
        while ii != ei:
                dst_data = &g[0].getData(g[0].getEdgeDst(ii), FLAG_UNPROTECTED)
                atomicMin[ComponentTy](dst_data.comp_current, src_data.comp_current.load())
                preincrement(ii)
##
# Label Propagation algorithm for 
# finding connected components
##
cdef void labelProp(Graph* graph):
    cdef:
        bool work_done = 1
        Timer T
    rounds = 0
    T.start()
    while(work_done):
        rounds += 1;
        with nogil:
            work_done = 0
            do_all(iterate(graph[0].begin(), graph[0].end()),
                     bind_leading(&labelPropOperator, graph, &work_done), 
                     no_pushes(),
                     steal(),
                     disable_conflict_detection(),
                     loopname("labelPropAlgo"))
    T.stop()
    gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")




#
# Main callsite for CC
#   
def connectedComponents(int numThreads, string filename):
    ## Hack: Need a better way to initialize shared
    ## memory runtime.
    sys = new SharedMemSys()
    cdef int new_numThreads = setActiveThreads(numThreads)
    gPrint(b"Running Pagerank on : ", filename, b"\n")
    if new_numThreads != numThreads:
        print("Warning, using fewer threads than requested")
    
    print("Using {0} thread(s).".format(new_numThreads))
    cdef Graph graph
    
    ## Read the CSR format of graph
    ## directly from disk.
    graph.readGraphFromGRFile(filename)
    
    initializeCompnents(&graph)
    labelProp(&graph)
    #printValuePR(&graph)
