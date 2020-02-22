#pragma once
#include "types.h"
#ifdef CPU_ONLY
#include "gtypes.h"
void update_all(Graph &g, const tensor_t &in, tensor_t &out, bool norm, const vec_t &norm_factor);
void update_all(Graph &g, const vec_t &in, tensor_t &out, bool norm, const vec_t &norm_factor);
#else
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#define TB_SIZE 256
#define WARP_SIZE 32
void update_all(CSRGraph &g, const float_t *in, float_t *out, bool norm, const float_t *norm_factor);
#endif

