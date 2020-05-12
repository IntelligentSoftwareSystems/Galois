#pragma once

#include "deepgalois/types.h"
#include "deepgalois/lgraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "graph_gpu.h"
#else
#include "galois/Galois.h"
#include "galois/graphs/NewGeneric.h"
#endif

namespace deepgalois {
using edge_iterator = index_t;
#ifdef __GALOIS_HET_CUDA__
using Graph         = CSRGraph;
using GraphGPU      = CSRGraph;
#else
using DGraph        = galois::graphs::DistGraph<char, void>;
using Graph         = LearningGraph;
#endif
} // namespace deepgalois
