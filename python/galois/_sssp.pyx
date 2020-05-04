# cython: cdivision= True
from galois.shmem cimport *
from cython.operator cimport preincrement, dereference as deref
from libstd.atomic cimport atomic

ctypedef uint32_t Dist
ctypedef atomic[Dist] AtomicDist
ctypedef atomic[uint32_t] atomuint32_t 

ctypedef uint32_t EdgeTy
ctypedef LC_CSR_Graph[AtomicDist, EdgeTy, dummy_true] Graph_CSR

# Cython bug: using a nested class from a previous typedef doesn't
# work for the time being. Instead, the full template specialization
# must be used to get the member type.
ctypedef LC_CSR_Graph[AtomicDist, EdgeTy, dummy_true].GraphNode GNodeCSR

cdef void printValue(Graph_CSR *g):
    cdef unsigned long numNodes = g[0].size()
    cdef AtomicDist *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        data = &g[0].getData(n)
        gPrint(b"\t", data[0].load(), b"\n")         
##############################################################################
## SSSP implementation
###########################################################################
#
# Initialization for SSSP
# Source distance is set to 0; Other nodes distance is set
# to number of nodes 
#
cdef void Initialize(Graph_CSR *g, unsigned long source):
    cdef:
        unsigned long numNodes = g[0].size()
        AtomicDist *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        #gPrint(n,"\n")
        data = &g[0].getData(n)
        if(n == source):
            data[0].store(0)
        else:
            data[0].store(numNodes)
        

ctypedef UpdateRequest[GNodeCSR, Dist] UpdateRequestObj
#
# SSSP Delta step Operator to be executed on each Graph node
#
cdef void ssspOperator(Graph_CSR *g, UpdateRequestObj item, UserContext[UpdateRequestObj] &ctx) nogil:

    cdef: 
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ei
        AtomicDist *src_data
        AtomicDist *dst_data
        Dist oldDist, newDist
        EdgeTy edge_data
        GNodeCSR dst
        unsigned long numNodes = g[0].size()
    
    src_data = &g[0].getData(item.src, FLAG_UNPROTECTED)    
    ii = g[0].edge_begin(item.src, FLAG_UNPROTECTED)
    ei = g[0].edge_end(item.src, FLAG_UNPROTECTED)
    if(src_data.load() < item.dist):
        return
    while ii != ei:
            dst = g[0].getEdgeDst(ii)
            dst_data = &g[0].getData(dst, FLAG_UNPROTECTED)
            edge_data = g[0].getEdgeData(ii, FLAG_UNPROTECTED)
            newDist = src_data[0].load() + edge_data

            oldDist = atomicMin[Dist](dst_data[0], newDist)
            if(newDist < oldDist):
                ctx.push(UpdateRequestObj(dst, newDist))

            preincrement(ii)

######
# SSSP Delta step algo using OBIM 
#####
ctypedef ChunkFIFO[Uint_64u] ChunkFIFO_64
ctypedef PerSocketChunkFIFO[Uint_64u] PerSocketChunkFIFO_64
ctypedef OrderedByIntegerMetric[UpdateRequestIndexer, PerSocketChunkFIFO_64] OBIM
cdef void ssspDeltaStep(Graph_CSR *graph, GNodeCSR source, uint32_t shift):
    cdef:
        Timer T
        InsertBag[UpdateRequestObj] initBag
        
    initBag.push(UpdateRequestObj(source, 0))
    T.start()
    with nogil:
        for_each(iterate(initBag),
                    bind_leading(&ssspOperator, graph),
                                wl[OBIM](UpdateRequestIndexer(shift)), 
                                #steal(), 
                                disable_conflict_detection(),
                                loopname("SSSP"))
    T.stop()
    gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")        



#######################
# Verification routines
#######################
cdef void not_visited_operator(Graph_CSR *graph, atomuint32_t *notVisited, GNodeCSR n):
    cdef: 
        AtomicDist *data
        uint32_t numNodes = graph[0].size()
    data = &graph[0].getData(n)
    if (data[0].load() >= numNodes):
        notVisited[0].fetch_add(1)

cdef void max_dist_operator(Graph_CSR *graph, GReduceMax[uint32_t] *maxDist , GNodeCSR n):
    cdef: 
        AtomicDist *data
        uint32_t numNodes = graph[0].size()
    data = &graph[0].getData(n)
    if(data[0].load() < numNodes):
        maxDist[0].update(data[0].load())

cdef bool verify_sssp(Graph_CSR *graph, GNodeCSR source):
    cdef: 
        atomuint32_t notVisited
        AtomicDist *data
        GReduceMax[uint32_t] maxDist;

    data = &graph[0].getData(source)
    if(data[0].load() is not 0):
        gPrint(b"ERROR: source has non-zero dist value == ", data[0].load(), b"\n")
    
    notVisited.store(0)
    with nogil:
        do_all(iterate(graph[0]),
                bind_leading(&not_visited_operator, graph, &notVisited), no_pushes(), steal(),
                loopname("not_visited_op"))

    if(notVisited.load() > 0):
        gPrint(notVisited.load(), b" unvisited nodes; this is an error if graph is strongly connected\n")

    with nogil:
        do_all(iterate(graph[0]),
                bind_leading(&max_dist_operator, graph, &maxDist), no_pushes(), steal(),
                loopname("not_visited_op"))

    gPrint(b"Max distance : ", maxDist.reduce(), b"\n")


#
# Main callsite for SSSP
#        
def sssp(int numThreads, uint32_t shift, unsigned long source, string filename):
    ## Hack: Need a better way to initialize shared
    ## memory runtime.
    sys = new SharedMemSys()
    cdef int new_numThreads = setActiveThreads(numThreads)
    if new_numThreads != numThreads:
        print("Warning, using fewer threads than requested")
    
    print("Using {0} thread(s).".format(new_numThreads))
    cdef Graph_CSR graph
    
    ## Read the CSR format of graph
    ## directly from disk.
    graph.readGraphFromGRFile(filename)
    gPrint(b"Using Source Node: ", source, b"\n");
    Initialize(&graph, source)
    #printValue(&graph)
    #ssspWorklist(&graph, <GNodeCSR>source)
    ssspDeltaStep(&graph, <GNodeCSR>source, shift)
    #verify_sssp(&graph, <GNodeCSR>source)
    gPrint(b"Node 1 has dist : ", graph.getData(1), b"\n")
    


