#pragma once

#include "deepgalois/types.h"
#include "deepgalois/lgraph.h"

#ifdef __GALOIS_HET_CUDA__
#define USE_CSRGRAPH
#ifdef USE_CSRGRAPH
#include "graph_gpu.h"
#endif
#else
#include "galois/Galois.h"
#include "galois/graphs/NewGeneric.h"
#endif

namespace deepgalois {
using edge_iterator = index_t;
using GraphCPU      = LearningGraph;
#ifdef __GALOIS_HET_CUDA__
using DGraph        = CSRGraph;
using Graph         = CSRGraph;
using GraphGPU      = CSRGraph;
#else
using DGraph        = galois::graphs::DistGraph<char, void>;
using Graph         = LearningGraph;
#endif
} // namespace deepgalois
