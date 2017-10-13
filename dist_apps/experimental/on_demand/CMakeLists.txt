include_directories(../../../include)
include_directories(${CMAKE_SOURCE_DIR}/libdist/include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

if(ENABLE_HETERO_GALOIS)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

function(cudaLibs name)
  cuda_add_library(${name}_cuda_v ${name}/gen_cuda.cu)
  target_link_libraries(${name}_cuda_v galois_gpu)
  # with bitset
  cuda_add_library(${name}_cuda_vbs ${name}/gen_cuda_bs.cu)
  target_link_libraries(${name}_cuda_vbs galois_gpu)
endfunction()

# GPU libraries created once then linked with correct version
if (ENABLE_HETERO_GALOIS)
  ##############################
  # bfs
  ##############################
  cudaLibs(bfs_pull)
  cudaLibs(bfs_push)
  ##############################
  # sssp
  ##############################
  cudaLibs(sssp_pull)
  cudaLibs(sssp_push)
  ##############################
  # cc
  ##############################
  cudaLibs(cc_pull)
  cudaLibs(cc_push)
  ##############################
  # pagerank
  ##############################
  cudaLibs(pagerank_pull)
  cudaLibs(pagerank_push)

  ##############################
  # kcore
  ##############################
  cudaLibs(kcore_push)
  # TODO kcore pull version?

  # TODO bc
  # TODO sgd
endif()

################################################################################
# Benchmark versions
################################################################################

function(versionGen name)
  app(${name}_v1 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v1 galois_dist distbench)
  target_compile_definitions(${name}_v1 PRIVATE __OPT_VERSION__=1)
  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
    target_link_libraries(${name}_v1 ${name}_cuda_v)
  endif()
  
  app(${name}_v2 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v2 galois_dist distbench)
  target_compile_definitions(${name}_v2 PRIVATE __OPT_VERSION__=2)
  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
    target_link_libraries(${name}_v2 ${name}_cuda_v)
  endif()
  
  app(${name}_v3 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v3 galois_dist distbench)
  target_compile_definitions(${name}_v3 PRIVATE __OPT_VERSION__=3)
  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
    target_link_libraries(${name}_v3 ${name}_cuda_vbs)
  endif()
  
  app(${name}_v4 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v4 galois_dist distbench)
  target_compile_definitions(${name}_v4 PRIVATE __OPT_VERSION__=4)
  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
    target_link_libraries(${name}_v4 ${name}_cuda_vbs)
  endif()

  app(${name}_v5 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v5 galois_dist distbench)
  target_compile_definitions(${name}_v5 PRIVATE __OPT_VERSION__=5)
  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_v5 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=5)
    target_link_libraries(${name}_v5 ${name}_cuda_vbs)
  endif()
endfunction()

function(versionGenNoGPU name)
  app(${name}_v1 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v1 galois_dist distbench)
  target_compile_definitions(${name}_v1 PRIVATE __OPT_VERSION__=1)
  
  app(${name}_v2 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v2 galois_dist distbench)
  target_compile_definitions(${name}_v2 PRIVATE __OPT_VERSION__=2)
  
  app(${name}_v3 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v3 galois_dist distbench)
  target_compile_definitions(${name}_v3 PRIVATE __OPT_VERSION__=3)
  
  app(${name}_v4 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v4 galois_dist distbench)
  target_compile_definitions(${name}_v4 PRIVATE __OPT_VERSION__=4)

  app(${name}_v5 ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name}_v5 galois_dist distbench)
  target_compile_definitions(${name}_v5 PRIVATE __OPT_VERSION__=5)
endfunction()

################################################################################
# BFS
################################################################################

versionGen(bfs_pull)
versionGen(bfs_push)

################################################################################
# SSSP
################################################################################

versionGen(sssp_pull)
versionGen(sssp_push)

################################################################################
# CC
################################################################################

versionGen(cc_pull)
versionGen(cc_push)

################################################################################
# PageRank
################################################################################

versionGen(pagerank_pull)
versionGen(pagerank_push)

################################################################################
# KCore
################################################################################

versionGen(kcore_push)
versionGenNoGPU(kcore_pull)

################################################################################
# BC
################################################################################

versionGenNoGPU(bc_pull)

# TODO BC, SGD, Kcore Pull (?)

#app(bc bc/gen.cpp DISTSAFE)
#target_link_libraries(bc galois_dist)
#if(ENABLE_HETERO_GALOIS)
#  target_compile_definitions(bc PRIVATE __GALOIS_HET_CUDA__=1)
#  target_link_libraries(bc bc_cuda)
#  cuda_add_library(bc_cuda bc/gen_cuda.cu)
#  target_link_libraries(bc_cuda ggrt)
#endif()
