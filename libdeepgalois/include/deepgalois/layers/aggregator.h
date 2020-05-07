#pragma once
#include "deepgalois/types.h"
//! For each node in the graph, add the embeddings of all of its neighbors
//! together (using norm_factor if specified)
#ifdef CPU_ONLY
#include "deepgalois/GraphTypes.h"
namespace deepgalois {
void update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, float_t* norm_factor);
void update_all_csrmm(size_t len, Graph& g, const float_t* in, float_t* out,
                      bool norm, float_t* norm_factor);
} // namespace deepgalois
#else
#include "deepgalois/GraphTypes.h"
//#include "graph_gpu.h"
namespace deepgalois {
void update_all(size_t len, GraphGPU& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
void update_all_csrmm(size_t len, GraphGPU& g, const float_t* in, float_t* out,
                      bool norm, const float_t* norm_factor);
} // namespace deepgalois
#endif
