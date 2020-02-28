#pragma once
#include <string>
#include <iostream>
typedef unsigned long long AccType;
void kcl_gpu_solver(std::string filename, unsigned k, AccType &total, size_t N_CHUNK = 1);
