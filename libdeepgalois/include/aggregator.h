#pragma once
#include "types.h"
#ifdef CPU_ONLY
#include "gtypes.h"
void update_all(size_t len, Graph &g, const float_t *in, float_t *out, bool norm, const float_t *norm_factor);
#else
#include "graph_gpu.h"
void update_all(size_t len, CSRGraph &g, const float_t *in, float_t *out, bool norm, const float_t *norm_factor);
#endif

