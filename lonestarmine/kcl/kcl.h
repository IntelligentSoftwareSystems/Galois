#pragma once
#include <string>
#include <iostream>
#include "pangolin/types.cuh"

void kcl_gpu_solver(std::string filename, unsigned k, AccType &total, size_t N_CHUNK = 1);
