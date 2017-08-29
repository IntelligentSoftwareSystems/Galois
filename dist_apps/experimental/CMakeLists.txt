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


