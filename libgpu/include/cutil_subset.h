/*
   cutil_subset.h

   Implements a subset of the CUDA utilities. Part of the GGC source code.

   TODO: actual owner copyright (NVIDIA) and license.
*/

#pragma once
#include "cub/cub.cuh"

#define CUDA_SAFE_CALL_NO_SYNC(call)                                           \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL_NO_SYNC(call);

#define CUDA_SAFE_THREAD_SYNC()                                                \
  {                                                                            \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
    }                                                                          \
  }

// from http://forums.nvidia.com/index.php?showtopic=186669
static __device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

static __device__ uint get_warpid(void) {
  uint ret;
  asm("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

// since cub::WarpScan doesn't work very well with disabled threads in the warp
__device__ __forceinline__ void warp_active_count(int& first, int& offset,
                                                  int& total) {
  unsigned int active = __ballot_sync(0xffffffff, 1);
  total               = __popc(active);
  offset              = __popc(active & cub::LaneMaskLt());
  first               = __ffs(active) - 1; // we know active != 0
}

// since cub::WarpScan doesn't work very well with disabled threads in the warp
__device__ __forceinline__ void
warp_active_count_zero_active(int& first, int& offset, int& total) {
  unsigned int active = __ballot_sync(0xffffffff, 1);
  total               = __popc(active);
  offset              = __popc(active & cub::LaneMaskLt());
  first               = 0;
}
