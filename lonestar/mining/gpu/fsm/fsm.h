#pragma once
#include <string>
#include <iostream>
#include "pangolin/types.cuh"

void fsm_gpu_solver(std::string fname, unsigned k, unsigned minsup,
                    AccType& total);
