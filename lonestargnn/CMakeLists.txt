include_directories(${CMAKE_SOURCE_DIR}/lonestargnn/include)
include_directories(${CMAKE_SOURCE_DIR}/libdeepgalois/include)
include_directories(${CUDA_HOME}/include)
if(ENABLE_HETERO_GALOIS)
  include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)
endif()

SET(BLAS_LIB_DIR ${OPENBLAS_ROOT}/lib64)
if(USE_MKL_BLAS)
  SET(BLAS_LIB_DIR "${MKL_ROOT}/lib/intel64")
endif()
link_directories(${BLAS_LIB_DIR})

if(NOT ENABLE_HETERO_GALOIS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
endif()

if(ENABLE_DIST_GALOIS)
  add_library(distgraphloader STATIC src/DistributedGraphLoader.cpp)
  target_include_directories(distgraphloader PUBLIC
    include
  )
  target_link_libraries(distgraphloader galois_cusp)
endif()

add_subdirectory(gcn)
