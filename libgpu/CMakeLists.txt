set(sources
  src/csr_graph.cu
  src/ggc_rt.cu
  moderngpu/src/mgpucontext.cu
  moderngpu/src/mgpuutil.cpp
)
add_library(galois_gpu ${sources})
target_compile_definitions(galois_gpu PRIVATE _FORCE_INLINES)
target_include_directories(galois_gpu PRIVATE moderngpu)
target_include_directories(galois_gpu PUBLIC include)
