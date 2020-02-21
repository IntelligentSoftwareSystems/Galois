#pragma once
#include "types.h"
#include "gtypes.h"

void update_all(Graph *g, const tensor_t &in, tensor_t &out, bool norm, const vec_t &norm_factor);
void update_all(Graph *g, const vec_t &in, tensor_t &out, bool norm, const vec_t &norm_factor);
