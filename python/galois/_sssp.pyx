# cython: cdivision= True
from galois.shmem cimport *
from cython.operator cimport preincrement, dereference as deref
from libstd.atomic cimport atomic

ctypedef atomic[uint32_t] atomuint32_t

ctypedef uint32_t EdgeTy
ctypedef LC_CSR_Graph[atomuint32_t, EdgeTy, dummy_true] Graph_CSR

# Cython bug: using a nested class from a previous typedef doesn't
# work for the time being. Instead, the full template specialization
# must be used to get the member type.
ctypedef LC_CSR_Graph[atomuint32_t, EdgeTy, dummy_true].GraphNode GNodeCSR

         
##############################################################################
## SSSP implementation
###########################################################################
#
# Initialization for BFS
#
cdef void Initialize(Graph_CSR *g, unsigned long source):
    cdef:
        unsigned long numNodes = g[0].size()
        atomuint32_t *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        #gPrint(n,"\n")
        data = &g[0].getData(n)
        if(n == source):
            data[0].store(0)
            gPrint(b"Srouce\n")
        else:
            data[0].store(numNodes)
        


#
# SSSP Operator to be executed on each Graph node
#
cdef void sssp_operator(Graph_CSR *g, GNodeCSR n, UserContext[GNodeCSR] &ctx) nogil:
    cdef: 
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ei
        atomuint32_t *src_data
        atomuint32_t *dst_data
        EdgeTy edge_data

        GNodeCSR dst
    src_data = &g[0].getData(n, FLAG_UNPROTECTED)    
    ii = g[0].edge_begin(n, FLAG_UNPROTECTED)
    ei = g[0].edge_end(n, FLAG_UNPROTECTED)
    while ii != ei:
            dst = g[0].getEdgeDst(ii)
            dst_data = &g[0].getData(dst, FLAG_UNPROTECTED)
            edge_data = g[0].getEdgeData(ii, FLAG_UNPROTECTED)
            preincrement(ii)


            
cdef void ssspDeltaStep(Graph_CSR *graph, GNodeCSR source):
    cdef:
        Timer T

    T.start()
    with nogil:
        for_each(iterate(graph[0].begin(), graph[0].end()),
                    bind_leading(&sssp_operator, graph), no_pushes(), steal(),
                    loopname("SSSP"))
    T.stop()
    gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")        
    
#
# Main callsite for Bfs
#        
def sssp(int numThreads, unsigned long source, string filename):
    cdef int new_numThreads = setActiveThreads(numThreads)
    gPrint(b"Hello this is gprint\n")
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
    #bfs_pull_topo(&graph)
    ssspDeltaStep(&graph, <GNodeCSR>source)
    #verify_sssp(&graph, <GNodeCSR>source)
    #gPrint("Node 1 has dist : ", graph.getData(1), "\n")
    


