add_library(galois_gpu)
add_library(Galois::gpu ALIAS galois_gpu)
add_dependencies(lib galois_gpu)

target_sources(galois_gpu PRIVATE
  src/csr_graph.cu
  src/ggc_rt.cu
  moderngpu/src/mgpucontext.cu
  moderngpu/src/mgpuutil.cpp
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/moderngpu/include>
  $<INSTALL_INTERFACE:include/moderngpu>
)

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cub>
  $<INSTALL_INTERFACE:include/cub>
)

target_compile_definitions(galois_gpu PRIVATE _FORCE_INLINES)
target_include_directories(galois_gpu PRIVATE moderngpu)
set_property(TARGET galois_gpu PROPERTY CUDA_STANDARD 14)
