if(ENABLE_HETERO_GALOIS)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

include_directories(../include)
include_directories(${CMAKE_SOURCE_DIR}/libdist/include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

function(distApp name)
  app(${name} ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name} galois_dist distbench)
  target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)

  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name} PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name} ${name}_cuda distbench)
    cuda_add_library(${name}_cuda ${name}/gen_cuda.cu)
    target_include_directories(${name}_cuda PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
    target_link_libraries(${name}_cuda ggrt distbench)
  endif()
endfunction()


distAppNoGPU(bfs_do)

distApp(bfs_push_wl)
distApp(sssp_push_wl)
distApp(pagerank_push_wl)
distApp(bfs_push_od)

distApp(pagerank_pull_od)

distAppNoGPU(bc_push)
distAppNoGPU(bc_pull)

distAppNoGPU(sgd_pull)

distAppNoGPU(t_bc)

distAppNoGPU(toy_read)
distAppNoGPU(toyinspect)

distAppNoGPU(graph_reader)

distAppNoGPU(transposecheck)
distAppNoGPU(weightcheck)
distAppNoGPU(samenesscheck)
distAppNoGPU(findhighdegree)
