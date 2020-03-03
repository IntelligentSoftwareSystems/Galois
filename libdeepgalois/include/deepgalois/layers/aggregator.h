#pragma once
#include "deepgalois/types.h"
#include "deepgalois/gtypes.h"
namespace deepgalois {
void update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
}
#ifndef CPU_ONLY
#include "graph_gpu.h"
namespace deepgalois {
void update_all(size_t len, CSRGraph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor);
void update_all_cusparse(size_t len, CSRGraph& g, const float_t* in, 
                float_t* out, bool norm, const float_t* norm_factor);
}
#endif
