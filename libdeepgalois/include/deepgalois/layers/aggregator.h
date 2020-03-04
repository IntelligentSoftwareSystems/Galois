#pragma once
#include "deepgalois/types.h"
//! For each node in the graph, add the embeddings of all of its neighbors
//! together (using norm_factor if specified)
#ifdef CPU_ONLY
#include "deepgalois/gtypes.h"
namespace deepgalois {
void update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
}
#else
#include "graph_gpu.h"
namespace deepgalois {
void update_all(size_t len, CSRGraph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
void update_all_cusparse(size_t len, CSRGraph& g, const float_t* in, 
                float_t* out, bool norm, const float_t* norm_factor);
}
#endif
