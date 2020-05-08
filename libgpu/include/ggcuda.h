/*
   ggcuda.h

   Implements GG CUDA runtime bits. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#pragma once

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
#define BLOCK_DIM_X blockDim.x

#define CONDITION enable_lb
#define MAX_INT 2147483647
#define THRESHOLD TOTAL_THREADS_1D
#define DEGREE_LIMIT ((CONDITION) ? (THRESHOLD) : (MAX_INT))
