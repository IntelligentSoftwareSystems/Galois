#pragma once
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

// TODO namespace

typedef galois::GAccumulator<acc_t> AccumF;
typedef galois::GAccumulator<size_t> AccumU;

#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
#endif

typedef Graph::GraphNode GNode;
