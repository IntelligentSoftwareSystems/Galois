set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/libgpu/moderngpu") # only required headers
include_directories("${MGPU_ROOT}/include")
set(sources
  src/csr_graph.cu
  src/ggc_rt.cu
  moderngpu/src/mgpucontext.cu
  moderngpu/src/mgpuutil.cpp
)
cuda_add_library(galois_gpu
  ${sources}
  OPTIONS -D_FORCE_INLINES
)
