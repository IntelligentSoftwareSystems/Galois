if(USE_GGC)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()


app(bfs_push-filter_wl bfs_push-filter_wl/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_wl galois_dist)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_wl PRIVATE __GALOIS_HET_CUDA__=1)
  target_link_libraries(bfs_push-filter_wl bfs_push-filter_cuda)
  cuda_add_library(bfs_push-filter_wl_cuda bfs_push-filter_wl/gen_cuda.cu)
  target_link_libraries(bfs_push-filter_wl_cuda ggrt)
endif()

app(bfs_push-filter_od bfs_push-filter_od/gen.cpp DISTSAFE)
target_link_libraries(bfs_push-filter_od galois_dist)
if(USE_GGC)
  target_compile_definitions(bfs_push-filter_od PRIVATE __GALOIS_HET_CUDA__=1)
  target_link_libraries(bfs_push-filter_od bfs_push-filter_cuda_od)
  cuda_add_library(bfs_push-filter_cuda_od bfs_push-filter_od/gen_cuda.cu)
  target_link_libraries(bfs_push-filter_cuda_od ggrt)
endif()

app(pagerank_pull_od pagerank_pull_od/gen.cpp DISTSAFE)
target_link_libraries(pagerank_pull_od galois_dist)
if(USE_GGC)
  target_compile_definitions(pagerank_pull_od PRIVATE __GALOIS_HET_CUDA__=1)
  target_link_libraries(pagerank_pull_od pagerank_pull_od_cuda)
  cuda_add_library(pagerank_pull_od_cuda pagerank_pull_od/gen_cuda.cu)
  target_link_libraries(pagerank_pull_od_cuda ggrt)
endif()

