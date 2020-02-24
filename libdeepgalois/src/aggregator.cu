#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "aggregator.h"
#include "math_functions.hh"

void update_all(size_t len, CSRGraph &g, const float_t *in, float_t *out, bool norm, const float_t *norm_factor) {
	unsigned n = g.nnodes;
	vadd_gpu(len, in, in, out);
}
	
