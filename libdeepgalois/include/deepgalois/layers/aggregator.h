#pragma once
//#include "deepgalois/types.h"
#include "deepgalois/gtypes.h"
//! For each node in the graph, add the embeddings of all of its neighbors
//! together (using norm_factor if specified)
#ifdef CPU_ONLY
namespace deepgalois {
void update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, float_t* norm_factor);
void update_all_csrmm(size_t len, Graph& g, const float_t* in, 
                float_t* out, bool norm, float_t* norm_factor);
}
#else
#include "graph_gpu.h"
namespace deepgalois {
void update_all(size_t len, GraphGPU& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
void update_all_csrmm(size_t len, GraphGPU& g, const float_t* in, 
                float_t* out, bool norm, const float_t* norm_factor);
}
#endif
