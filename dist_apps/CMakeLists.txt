add_library(distbench STATIC src/DistBenchStart.cpp)
target_include_directories(distbench PUBLIC 
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# turn on cuda for distbench as well
if(USE_GGC)
  target_compile_definitions(distbench PRIVATE __GALOIS_HET_CUDA__=1)
endif()

target_link_libraries(distbench galois_shmem galois_dist gllvm)

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/libdist/include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

if(USE_GGC)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

function(distApp name)
  app(${name} ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name} galois_dist distbench)
  target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()

  if(USE_GGC)
    target_compile_definitions(${name} PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name} ${name}_cuda distbench)
    cuda_add_library(${name}_cuda ${name}/gen_cuda.cu)
    target_include_directories(${name}_cuda PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
    target_link_libraries(${name}_cuda ggrt distbench)
  endif()
endfunction()

function(distAppNoGPU name)
  app(${name} ${name}/gen.cpp DISTSAFE)
  target_link_libraries(${name} galois_dist distbench)
  target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()
endfunction()

distApp(bfs_pull)
distApp(bfs_push)

distApp(cc_pull)
distApp(cc_push)

distApp(pagerank_pull)
distApp(pagerank_push)

distApp(sssp_pull)
distApp(sssp_push)

distApp(kcore_push)
distAppNoGPU(kcore_pull)
