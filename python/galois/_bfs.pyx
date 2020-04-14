# cython: cdivision = True

from galois.shmem cimport *
from cython.operator cimport preincrement, dereference as deref

ctypedef atomic[uint32_t] atomuint32_t

ctypedef LC_CSR_Graph[uint32_t, void, dummy_true] Graph_CSR

# Cython bug: using a nested class from a previous typedef doesn't
# work for the time being. Instead, the full template specialization
# must be used to get the member type.
ctypedef LC_CSR_Graph[uint32_t, void, dummy_true].GraphNode GNodeCSR

cdef void printValue(Graph_CSR *g):
    cdef unsigned long numNodes = g[0].size()
    cdef uint32_t *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        data = &g[0].getData(n)
        gPrint(b"\t", data[0], b"\n")
         
##############################################################################
## Bfs implementation
###########################################################################
#
# Initialization for BFS
#
cdef void Initialize(Graph_CSR *g, unsigned long source):
    cdef unsigned long numNodes = g[0].size()
    cdef: 
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ei
        uint32_t *data
    gPrint(b"Number of nodes : ", numNodes, b"\n")
    for n in range(numNodes):
        #gPrint(n,"\n")
        data = &g[0].getData(n)
        if(n == source):
            data[0] = 0
            gPrint(b"Srouce\n")
        else:
            data[0] = numNodes
        

#
# BFS Operator to be executed on each Graph node
#
cdef void bfs_operator(Graph_CSR *g, bool *work_done, GNodeCSR n, UserContext[GNodeCSR] &ctx) nogil:
    cdef: 
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ei
        uint32_t *src_data
        uint32_t *dst_data
    src_data = &g[0].getData(n)    
    ii = g[0].edge_begin(n)
    ei = g[0].edge_end(n)
    while ii != ei:
            dst_data = &g[0].getData(g[0].getEdgeDst(ii))
            if(src_data[0] > dst_data[0] + 1):
                src_data[0] = dst_data[0] + 1
                work_done[0] = 1
            preincrement(ii)
            
cdef void bfs_pull_topo(Graph_CSR *graph):
    cdef bool work_done = 1
    cdef Timer T
    rounds = 0; 
    while(work_done):
        rounds += 1;
        print("starting for_each")
        gPrint(b"Work done Before : ", work_done, b"\n")
        with nogil:
            T.start()
            work_done = 0
            for_each(iterate(graph[0].begin(), graph[0].end()),
                     bind_leading(&bfs_operator, graph, &work_done), no_pushes())#,
                     #loopname("name1"))
            T.stop()
            gPrint(b"Work done : ", work_done, b"\n")
            gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")
    print("Number of rounds : ", rounds, "\n")


#
# BFS sync operator to be executed on each Graph node
#
cdef void bfs_sync_operator(Graph_CSR *g, InsertBag[GNodeCSR] *next, int nextLevel, GNodeCSR n) nogil:
    cdef: 
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ii
        LC_CSR_Graph[uint32_t, void, dummy_true].edge_iterator ei
        uint32_t *src_data
        uint32_t *dst_data
        uint32_t numNodes = g[0].size()
        GNodeCSR dst
    src_data = &g[0].getData(n)    
    ii = g[0].edge_begin(n)
    ei = g[0].edge_end(n)
    while ii != ei:
            dst = g[0].getEdgeDst(ii)
            dst_data = &g[0].getData(dst)
            if(dst_data[0] == numNodes):
                dst_data[0] = nextLevel
                next[0].push(dst)
            preincrement(ii)
            
cdef void bfs_sync(Graph_CSR *graph, GNodeCSR source):
    cdef:
        Timer T
        InsertBag[GNodeCSR] curr, next
        uint32_t nextLevel = 0;
    
    next.push(source)
    T.start()
    while(not next.empty()):
        curr.swap(next)
        next.clear()
        nextLevel += 1;
        with nogil:
            do_all(iterate(curr),
                     bind_leading(&bfs_sync_operator, graph, &next, nextLevel), no_pushes(), steal(),
                     loopname("bfs_sync"))
    T.stop()
    gPrint(b"Elapsed time:", T.get(), b" milliseconds.\n")        
    print("Number of rounds : ", nextLevel, "\n")

cdef void not_visited_operator(Graph_CSR *graph, atomuint32_t *notVisited, GNodeCSR n):
    cdef: 
        uint32_t *data
        uint32_t numNodes = graph[0].size()
    data = &graph[0].getData(n)
    #if (data[0] >= numNodes):
        #notVisited[0].fetch_add(1)

cdef void max_dist_operator(Graph_CSR *graph, GReduceMax[uint32_t] *maxDist , GNodeCSR n):
    cdef: 
        uint32_t *data
        uint32_t numNodes = graph[0].size()
    data = &graph[0].getData(n)
    if(data[0] < numNodes):
        maxDist[0].update(data[0])


cdef bool verify_bfs(Graph_CSR *graph, GNodeCSR source):
    cdef: 
        atomuint32_t notVisited
        uint32_t *data
        GReduceMax[uint32_t] maxDist;

    data = &graph[0].getData(source)
    if(data[0] is not 0):
        gPrint(b"ERROR: source has non-zero dist value == ", data[0], b"\n")
    
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
# Main callsite for Bfs
#        
def bfs(int numThreads, unsigned long source, string filename):
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
    bfs_sync(&graph, <GNodeCSR>source)
    verify_bfs(&graph, <GNodeCSR>source)
    gPrint(b"Node 1 has dist : ", graph.getData(1), b"\n")
    


