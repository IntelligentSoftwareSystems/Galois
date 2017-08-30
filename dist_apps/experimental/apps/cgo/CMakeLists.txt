if(USE_GGC)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

# GGC libraries created once then linked with correct version
if (USE_GGC)
  ##############################
  # bfs
  ##############################
  cuda_add_library(bfs_pull-topological_cuda_v bfs_pull-topological/gen_cuda.cu)
  target_link_libraries(bfs_pull-topological_cuda_v ggrt)
  cuda_add_library(bfs_push-filter_cuda_v bfs_push-filter/gen_cuda.cu)
  target_link_libraries(bfs_push-filter_cuda_v ggrt)

  # with bitset
  cuda_add_library(bfs_pull-topological_cuda_vbs bfs_pull-topological/gen_cuda_bs.cu)
  target_link_libraries(bfs_pull-topological_cuda_vbs ggrt)
  cuda_add_library(bfs_push-filter_cuda_vbs bfs_push-filter/gen_cuda_bs.cu)
  target_link_libraries(bfs_push-filter_cuda_vbs ggrt)

  ##############################
  # sssp
  ##############################
  cuda_add_library(sssp_pull-topological_cuda_v sssp_pull-topological/gen_cuda.cu)
  target_link_libraries(sssp_pull-topological_cuda_v ggrt)
  cuda_add_library(sssp_push-filter_cuda_v sssp_push-filter/gen_cuda.cu)
  target_link_libraries(sssp_push-filter_cuda_v ggrt)

  # with bitset
  cuda_add_library(sssp_pull-topological_cuda_vbs sssp_pull-topological/gen_cuda_bs.cu)
  target_link_libraries(sssp_pull-topological_cuda_vbs ggrt)
  cuda_add_library(sssp_push-filter_cuda_vbs sssp_push-filter/gen_cuda_bs.cu)
  target_link_libraries(sssp_push-filter_cuda_vbs ggrt)

  ##############################
  # cc
  ##############################
  cuda_add_library(cc_pull-topological_cuda_v cc_pull-topological/gen_cuda.cu)
  target_link_libraries(cc_pull-topological_cuda_v ggrt)
  cuda_add_library(cc_push-filter_cuda_v cc_push-filter/gen_cuda.cu)
  target_link_libraries(cc_push-filter_cuda_v ggrt)

  # with bitset
  cuda_add_library(cc_pull-topological_cuda_vbs cc_pull-topological/gen_cuda_bs.cu)
  target_link_libraries(cc_pull-topological_cuda_vbs ggrt)
  cuda_add_library(cc_push-filter_cuda_vbs cc_push-filter/gen_cuda_bs.cu)
  target_link_libraries(cc_push-filter_cuda_vbs ggrt)

  ##############################
  # pagerank
  ##############################
  # TODO bitset version
  cuda_add_library(pagerank_pull-topological_cuda_v pagerank_pull-topological/gen_cuda.cu)
  target_link_libraries(pagerank_pull-topological_cuda_v ggrt)
  cuda_add_library(pagerank_push-filter_cuda_v pagerank_push-filter/gen_cuda.cu)
  target_link_libraries(pagerank_push-filter_cuda_v ggrt)

  ##############################
  # kcore
  ##############################
  # TODO bitset version
  cuda_add_library(kcore_push-filter_cuda_v kcore_push-filter/gen_cuda.cu)
  target_link_libraries(kcore_push-filter_cuda_v ggrt)

  # TODO kcore pull version?
  # TODO bc
  # TODO sgd
endif()

################################################################################
# BFS
################################################################################

app(bfs_pull-topological_v1 bfs_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(bfs_pull-topological_v1 galois_dist)
target_compile_definitions(bfs_pull-topological_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(bfs_pull-topological_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(bfs_pull-topological_v1 bfs_pull-topological_cuda_v)
endif()

app(bfs_pull-topological_v2 bfs_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(bfs_pull-topological_v2 galois_dist)
target_compile_definitions(bfs_pull-topological_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(bfs_pull-topological_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(bfs_pull-topological_v2 bfs_pull-topological_cuda_v)
endif()

app(bfs_pull-topological_v3 bfs_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(bfs_pull-topological_v3 galois_dist)
target_compile_definitions(bfs_pull-topological_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(bfs_pull-topological_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(bfs_pull-topological_v3 bfs_pull-topological_cuda_vbs)
endif()

app(bfs_pull-topological_v4 bfs_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(bfs_pull-topological_v4 galois_dist)
target_compile_definitions(bfs_pull-topological_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(bfs_pull-topological_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(bfs_pull-topological_v4 bfs_pull-topological_cuda_vbs)
endif()

app(bfs_push-filter_v1 bfs_push-filter/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_v1 galois_dist)
target_compile_definitions(bfs_push-filter_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(bfs_push-filter_v1 bfs_push-filter_cuda_v)
endif()

app(bfs_push-filter_v2 bfs_push-filter/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_v2 galois_dist)
target_compile_definitions(bfs_push-filter_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(bfs_push-filter_v2 bfs_push-filter_cuda_v)
endif()

app(bfs_push-filter_v3 bfs_push-filter/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_v3 galois_dist)
target_compile_definitions(bfs_push-filter_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(bfs_push-filter_v3 bfs_push-filter_cuda_vbs)
endif()

app(bfs_push-filter_v4 bfs_push-filter/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_v4 galois_dist)
target_compile_definitions(bfs_push-filter_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(bfs_push-filter_v4 bfs_push-filter_cuda_vbs)
endif()

################################################################################
# SSSP
################################################################################

app(sssp_pull-topological_v1 sssp_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(sssp_pull-topological_v1 galois_dist)
target_compile_definitions(sssp_pull-topological_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(sssp_pull-topological_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(sssp_pull-topological_v1 sssp_pull-topological_cuda_v)
endif()

app(sssp_pull-topological_v2 sssp_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(sssp_pull-topological_v2 galois_dist)
target_compile_definitions(sssp_pull-topological_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(sssp_pull-topological_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(sssp_pull-topological_v2 sssp_pull-topological_cuda_v)
endif()

app(sssp_pull-topological_v3 sssp_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(sssp_pull-topological_v3 galois_dist)
target_compile_definitions(sssp_pull-topological_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(sssp_pull-topological_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(sssp_pull-topological_v3 sssp_pull-topological_cuda_vbs)
endif()

app(sssp_pull-topological_v4 sssp_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(sssp_pull-topological_v4 galois_dist)
target_compile_definitions(sssp_pull-topological_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(sssp_pull-topological_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(sssp_pull-topological_v4 sssp_pull-topological_cuda_vbs)
endif()

app(sssp_push-filter_v1 sssp_push-filter/gen.cpp DISTSAFE)
target_link_libraries(sssp_push-filter_v1 galois_dist)
target_compile_definitions(sssp_push-filter_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(sssp_push-filter_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(sssp_push-filter_v1 sssp_push-filter_cuda_v)
endif()

app(sssp_push-filter_v2 sssp_push-filter/gen.cpp DISTSAFE)
target_link_libraries(sssp_push-filter_v2 galois_dist)
target_compile_definitions(sssp_push-filter_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(sssp_push-filter_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(sssp_push-filter_v2 sssp_push-filter_cuda_v)
endif()

app(sssp_push-filter_v3 sssp_push-filter/gen.cpp DISTSAFE)
target_link_libraries(sssp_push-filter_v3 galois_dist)
target_compile_definitions(sssp_push-filter_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(sssp_push-filter_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(sssp_push-filter_v3 sssp_push-filter_cuda_vbs)
endif()

app(sssp_push-filter_v4 sssp_push-filter/gen.cpp DISTSAFE)
target_link_libraries(sssp_push-filter_v4 galois_dist)
target_compile_definitions(sssp_push-filter_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(sssp_push-filter_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(sssp_push-filter_v4 sssp_push-filter_cuda_vbs)
endif()


################################################################################
# CC
################################################################################

app(cc_pull-topological_v1 cc_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(cc_pull-topological_v1 galois_dist)
target_compile_definitions(cc_pull-topological_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(cc_pull-topological_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(cc_pull-topological_v1 cc_pull-topological_cuda_v)
endif()

app(cc_pull-topological_v2 cc_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(cc_pull-topological_v2 galois_dist)
target_compile_definitions(cc_pull-topological_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(cc_pull-topological_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(cc_pull-topological_v2 cc_pull-topological_cuda_v)
endif()

app(cc_pull-topological_v3 cc_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(cc_pull-topological_v3 galois_dist)
target_compile_definitions(cc_pull-topological_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(cc_pull-topological_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(cc_pull-topological_v3 cc_pull-topological_cuda_vbs)
endif()

app(cc_pull-topological_v4 cc_pull-topological/gen.cpp DISTSAFE)
target_link_libraries(cc_pull-topological_v4 galois_dist)
target_compile_definitions(cc_pull-topological_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(cc_pull-topological_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(cc_pull-topological_v4 cc_pull-topological_cuda_vbs)
endif()

app(cc_push-filter_v1 cc_push-filter/gen.cpp DISTSAFE)
target_link_libraries(cc_push-filter_v1 galois_dist)
target_compile_definitions(cc_push-filter_v1 PRIVATE __OPT_VERSION__=1)
if(USE_GGC)
  target_compile_definitions(cc_push-filter_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
  target_link_libraries(cc_push-filter_v1 cc_push-filter_cuda_v)
endif()

app(cc_push-filter_v2 cc_push-filter/gen.cpp DISTSAFE)
target_link_libraries(cc_push-filter_v2 galois_dist)
target_compile_definitions(cc_push-filter_v2 PRIVATE __OPT_VERSION__=2)
if(USE_GGC)
  target_compile_definitions(cc_push-filter_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
  target_link_libraries(cc_push-filter_v2 cc_push-filter_cuda_v)
endif()

app(cc_push-filter_v3 cc_push-filter/gen.cpp DISTSAFE)
target_link_libraries(cc_push-filter_v3 galois_dist)
target_compile_definitions(cc_push-filter_v3 PRIVATE __OPT_VERSION__=3)
if(USE_GGC)
  target_compile_definitions(cc_push-filter_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
  target_link_libraries(cc_push-filter_v3 cc_push-filter_cuda_vbs)
endif()

app(cc_push-filter_v4 cc_push-filter/gen.cpp DISTSAFE)
target_link_libraries(cc_push-filter_v4 galois_dist)
target_compile_definitions(cc_push-filter_v4 PRIVATE __OPT_VERSION__=4)
if(USE_GGC)
  target_compile_definitions(cc_push-filter_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
  target_link_libraries(cc_push-filter_v4 cc_push-filter_cuda_vbs)
endif()


################################################################################
# PageRank
################################################################################

#app(pagerank_pull-topological_v1 pagerank_pull-topological/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_pull-topological_v1 galois_dist)
#target_compile_definitions(pagerank_pull-topological_v1 PRIVATE __OPT_VERSION__=1)
#if(USE_GGC)
#  target_compile_definitions(pagerank_pull-topological_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
#  target_link_libraries(pagerank_pull-topological_v1 pagerank_pull-topological_cuda_v)
#endif()
#
#app(pagerank_pull-topological_v2 pagerank_pull-topological/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_pull-topological_v2 galois_dist)
#target_compile_definitions(pagerank_pull-topological_v2 PRIVATE __OPT_VERSION__=2)
#if(USE_GGC)
#  target_compile_definitions(pagerank_pull-topological_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
#  target_link_libraries(pagerank_pull-topological_v2 pagerank_pull-topological_cuda_v)
#endif()
#
#app(pagerank_pull-topological_v3 pagerank_pull-topological/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_pull-topological_v3 galois_dist)
#target_compile_definitions(pagerank_pull-topological_v3 PRIVATE __OPT_VERSION__=3)
#if(USE_GGC)
#  target_compile_definitions(pagerank_pull-topological_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
#  target_link_libraries(pagerank_pull-topological_v3 pagerank_pull-topological_cuda_v)
#endif()
#
#app(pagerank_pull-topological_v4 pagerank_pull-topological/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_pull-topological_v4 galois_dist)
#target_compile_definitions(pagerank_pull-topological_v4 PRIVATE __OPT_VERSION__=4)
#if(USE_GGC)
#  target_compile_definitions(pagerank_pull-topological_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
#  target_link_libraries(pagerank_pull-topological_v4 pagerank_pull-topological_cuda_v)
#endif()
#
#app(pagerank_push-filter_v1 pagerank_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_push-filter_v1 galois_dist)
#target_compile_definitions(pagerank_push-filter_v1 PRIVATE __OPT_VERSION__=1)
#if(USE_GGC)
#  target_compile_definitions(pagerank_push-filter_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
#  target_link_libraries(pagerank_push-filter_v1 pagerank_push-filter_cuda_v)
#endif()
#
#app(pagerank_push-filter_v2 pagerank_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_push-filter_v2 galois_dist)
#target_compile_definitions(pagerank_push-filter_v2 PRIVATE __OPT_VERSION__=2)
#if(USE_GGC)
#  target_compile_definitions(pagerank_push-filter_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
#  target_link_libraries(pagerank_push-filter_v2 pagerank_push-filter_cuda_v)
#endif()
#
#app(pagerank_push-filter_v3 pagerank_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_push-filter_v3 galois_dist)
#target_compile_definitions(pagerank_push-filter_v3 PRIVATE __OPT_VERSION__=3)
#if(USE_GGC)
#  target_compile_definitions(pagerank_push-filter_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
#  target_link_libraries(pagerank_push-filter_v3 pagerank_push-filter_cuda_v)
#endif()
#
#app(pagerank_push-filter_v4 pagerank_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(pagerank_push-filter_v4 galois_dist)
#target_compile_definitions(pagerank_push-filter_v4 PRIVATE __OPT_VERSION__=4)
#if(USE_GGC)
#  target_compile_definitions(pagerank_push-filter_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
#  target_link_libraries(pagerank_push-filter_v4 pagerank_push-filter_cuda_v)
#endif()

################################################################################
# KCore
################################################################################

#app(kcore_push-filter_v1 kcore_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(kcore_push-filter_v1 galois_dist)
#target_compile_definitions(kcore_push-filter_v1 PRIVATE __OPT_VERSION__=1)
#if(USE_GGC)
#  target_compile_definitions(kcore_push-filter_v1 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=1)
#  target_link_libraries(kcore_push-filter_v1 kcore_push-filter_cuda_v)
#endif()
#
#app(kcore_push-filter_v2 kcore_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(kcore_push-filter_v2 galois_dist)
#target_compile_definitions(kcore_push-filter_v2 PRIVATE __OPT_VERSION__=2)
#if(USE_GGC)
#  target_compile_definitions(kcore_push-filter_v2 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=2)
#  target_link_libraries(kcore_push-filter_v2 kcore_push-filter_cuda_v)
#endif()
#
#app(kcore_push-filter_v3 kcore_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(kcore_push-filter_v3 galois_dist)
#target_compile_definitions(kcore_push-filter_v3 PRIVATE __OPT_VERSION__=3)
#if(USE_GGC)
#  target_compile_definitions(kcore_push-filter_v3 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=3)
#  target_link_libraries(kcore_push-filter_v3 kcore_push-filter_cuda_v)
#endif()
#
#app(kcore_push-filter_v4 kcore_push-filter/gen.cpp DISTSAFE)
#target_link_libraries(kcore_push-filter_v4 galois_dist)
#target_compile_definitions(kcore_push-filter_v4 PRIVATE __OPT_VERSION__=4)
#if(USE_GGC)
#  target_compile_definitions(kcore_push-filter_v4 PRIVATE __GALOIS_HET_CUDA__=1 __OPT_VERSION__=4)
#  target_link_libraries(kcore_push-filter_v4 kcore_push-filter_cuda_v)
#endif()


# TODO BC, SGD, Kcore Pull (?)

#app(bc bc/gen.cpp DISTSAFE)
#target_link_libraries(bc galois_dist)
#if(USE_GGC)
#  target_compile_definitions(bc PRIVATE __GALOIS_HET_CUDA__=1)
#  target_link_libraries(bc bc_cuda)
#  cuda_add_library(bc_cuda bc/gen_cuda.cu)
#  target_link_libraries(bc_cuda ggrt)
#endif()
