#ifndef CUTIL_SUBSET_H
#define CUTIL_SUBSET_H

#define CUDA_SAFE_CALL_NO_SYNC(call)                                           \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",  \
              err, __FILE__, __LINE__, cudaGetErrorString(err));               \
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
static __device__ unsigned get_smid(void) {
  unsigned ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

inline unsigned CudaTest(const char* msg) {
  cudaError_t e;
  // cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
    // return 1;
  }
  return 0;
}
#endif
