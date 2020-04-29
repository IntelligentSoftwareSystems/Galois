#ifndef __DG_GTYPES__
#define __DG_GTYPES__

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "deepgalois/types.h"
#ifdef GALOIS_USE_DIST
#include "galois/graphs/NewGeneric.h"
#endif

// TODO namespace

typedef galois::GAccumulator<acc_t> AccumF;
typedef galois::GAccumulator<size_t> AccumU;
#ifdef GALOIS_USE_DIST
using AccuracyAccum = galois::DGAccumulator<acc_t>;
#endif

#ifndef GALOIS_USE_DIST
#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<void, void, false, false, false, void, uint64_t, uint64_t>::
    with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif
#else
using Graph = galois::graphs::DistGraph<char, void>;
#endif

typedef Graph::GraphNode GNode;

#endif
