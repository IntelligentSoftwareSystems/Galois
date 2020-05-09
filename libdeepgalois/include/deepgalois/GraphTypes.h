#pragma once

#include "deepgalois/types.h"
#include "galois/Galois.h"
#include "galois/graphs/NewGeneric.h"
#include "deepgalois/lgraph.h"

#ifdef __GALOIS_HET_CUDA__
// TODO reintroduce GPU as necessary here
#endif

namespace deepgalois {
using edge_iterator = index_t;
using DGraph        = galois::graphs::DistGraph<char, void>;
using Graph         = LearningGraph;
} // namespace deepgalois
