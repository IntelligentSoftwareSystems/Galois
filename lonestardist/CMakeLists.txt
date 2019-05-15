# NOTE: this is required because 
# target_link_libraries() does not work with cuda_add_library() 
# (in old versions of CMake)
include_directories(${CMAKE_SOURCE_DIR}/libgluon/include)

add_library(distbench STATIC src/DistBenchStart.cpp src/DistributedGraphLoader.cpp)
target_include_directories(distbench PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(distbench galois_cusp galois_gluon)

if(ENABLE_HETERO_GALOIS)
  # turn on cuda for distbench as well
  target_compile_definitions(distbench PRIVATE __GALOIS_HET_CUDA__=1)

  # for debugging
  add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

# CuSP uses DGAccumulator which uses LCI if enabled
if (USE_LCI)
  target_include_directories(distbench PUBLIC ${LWCI_INCLUDE})
endif()

function(distApp name)
  FILE(GLOB CPPSOURCES ${name}*.cpp)
  app(${name} ${CPPSOURCES} DISTSAFE)
  target_link_libraries(${name} distbench)
  if(REPORT_PER_ROUND_STATS)
    target_compile_definitions(${name} PRIVATE DIST_PER_ROUND_TIMER=1)
  endif()
  if(REPORT_COMM_STATS)
    target_compile_definitions(${name} PRIVATE MORE_COMM_STATS=1)
  endif()
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()

  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name} PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name} ${name}_cuda)

    FILE(GLOB CUSOURCES ${name}*.cu)
    cuda_add_library(${name}_cuda ${CUSOURCES})
    target_link_libraries(${name}_cuda galois_gpu)
  endif()
endfunction()

function(distAppNoGPU name)
  FILE(GLOB CPPSOURCES ${name}*.cpp)
  app(${name} ${CPPSOURCES} DISTSAFE)
  target_link_libraries(${name} distbench)
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()
endfunction()

add_subdirectory(bc)
add_subdirectory(bfs)
add_subdirectory(cc)
add_subdirectory(kcore)
add_subdirectory(pagerank)
add_subdirectory(sgd)
add_subdirectory(sssp)

add_subdirectory(partition)
