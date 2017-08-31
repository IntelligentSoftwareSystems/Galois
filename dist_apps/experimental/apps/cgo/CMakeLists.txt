if(USE_GGC)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

function(cudaLibs name)
  cuda_add_library(${name}_cuda_v ${name}/gen_cuda.cu)
  target_link_libraries(${name}_cuda_v ggrt)
  # with bitset
  cuda_add_library(${name}_cuda_vbs ${name}/gen_cuda_bs.cu)
  target_link_libraries(${name}_cuda_vbs ggrt)
endfunction()

# GGC libraries created once then linked with correct version
if (USE_GGC)
  ##############################
  # bfs
  ##############################
  cudaLibs(bfs_pull-topological)
  cudaLibs(bfs_push-filter)
  ##############################
  # sssp
  ##############################
  cudaLibs(sssp_pull-topological)
  cudaLibs(sssp_push-filter)
  ##############################
  # cc
  ##############################
  cudaLibs(cc_pull-topological)
  cudaLibs(cc_push-filter)
  ##############################
  # pagerank
  ##############################
  cudaLibs(pagerank_pull-topological)
  cudaLibs(pagerank_push-filter)

  ##############################
  # kcore
  ##############################
  cudaLibs(kcore_push-filter)
  # TODO kcore pull version?

  # TODO bc
  # TODO sgd
endif()

################################################################################
# Benchmark versions
################################################################################

function(versionGen name)
  app(${name}_v1 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v1 galois_dist)
  target_compile_definitions(${name}_v1 PRIVATE __OPT_VERSION__=1)
  if(USE_GGC)
    target_compile_definitions(${name}_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
    target_link_libraries(${name}_v1 ${name}_cuda_v)
  endif()
  
  app(${name}_v2 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v2 galois_dist)
  target_compile_definitions(${name}_v2 PRIVATE __OPT_VERSION__=2)
  if(USE_GGC)
    target_compile_definitions(${name}_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
    target_link_libraries(${name}_v2 ${name}_cuda_v)
  endif()
  
  app(${name}_v3 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v3 galois_dist)
  target_compile_definitions(${name}_v3 PRIVATE __OPT_VERSION__=3)
  if(USE_GGC)
    target_compile_definitions(${name}_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
    target_link_libraries(${name}_v3 ${name}_cuda_vbs)
  endif()
  
  app(${name}_v4 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v4 galois_dist)
  target_compile_definitions(${name}_v4 PRIVATE __OPT_VERSION__=4)
  if(USE_GGC)
    target_compile_definitions(${name}_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
    target_link_libraries(${name}_v4 ${name}_cuda_vbs)
  endif()
endfunction()

################################################################################
# BFS
################################################################################

versionGen(bfs_pull-topological)
versionGen(bfs_push-filter)

################################################################################
# SSSP
################################################################################

versionGen(sssp_pull-topological)
versionGen(sssp_push-filter)

################################################################################
# CC
################################################################################

versionGen(cc_pull-topological)
versionGen(cc_push-filter)

################################################################################
# PageRank
################################################################################

versionGen(pagerank_pull-topological)
versionGen(pagerank_push-filter)

################################################################################
# KCore
################################################################################

versionGen(kcore_push-filter)


# TODO BC, SGD, Kcore Pull (?)

#app(bc bc/gen.cpp DISTSAFE)
#target_link_libraries(bc galois_dist)
#if(USE_GGC)
#  target_compile_definitions(bc PRIVATE __GALOIS_HET_CUDA__=1)
#  target_link_libraries(bc bc_cuda)
#  cuda_add_library(bc_cuda bc/gen_cuda.cu)
#  target_link_libraries(bc_cuda ggrt)
#endif()
