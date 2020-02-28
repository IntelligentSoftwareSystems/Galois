#pragma once
#include <string>
#include <iostream>
#include "types.cuh"

void tc_gpu_solver(std::string filename, AccType &total, size_t N_CHUNK = 1);
