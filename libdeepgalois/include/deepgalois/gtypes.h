#pragma once
#define USE_CSRGRAPH

#include "deepgalois/types.h"
#ifdef GALOIS_USE_DIST
#include "galois/Galois.h"
#include "galois/graphs/NewGeneric.h"
#else
#ifdef CPU_ONLY
//#include "galois/Galois.h"
//#include "galois/graphs/LCGraph.h"
#include "deepgalois/lgraph.h"
#else
#ifdef USE_CSRGRAPH
#include "deepgalois/lgraph.h"
#include "graph_gpu.h"
#else
#include "deepgalois/lgraph.h"
#endif
#endif
#endif

#ifndef GALOIS_USE_DIST

namespace deepgalois {
typedef index_t edge_iterator;
//#ifdef EDGE_LABEL
// typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::
//    with_numa_alloc<true>::type ::with_no_lockable<true>::type LCGraph;
//#else
// typedef galois::graphs::LC_CSR_Graph<void, void, false, false, false, void,
// uint64_t, uint64_t>::
//    with_numa_alloc<true>::type ::with_no_lockable<true>::type LCGraph;
//#endif
// typedef LCGraph Graph;
// typedef Graph::edge_iterator edge_iterator;
typedef LearningGraph Graph;
#ifdef USE_CSRGRAPH
typedef CSRGraph GraphGPU;
#else
typedef LearningGraph GraphGPU;
#endif
} // namespace deepgalois

#else

namespace deepgalois {
// TODO check if this needs changing
typedef index_t edge_iterator;
using Graph = galois::graphs::DistGraph<char, void>;
} // namespace deepgalois

#endif
