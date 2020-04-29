#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "pangolin/types.cuh"

void motif_gpu_solver(std::string fname, unsigned k, std::vector<AccType> &acc, size_t N_CHUNK=1);
