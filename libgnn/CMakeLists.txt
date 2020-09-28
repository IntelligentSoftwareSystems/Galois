set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
SET(BLAS_INC_DIR ${OPENBLAS_ROOT}/include/openblas)
SET(BLAS_LIB_DIR ${OPENBLAS_ROOT}/lib64)
set(BLAS_LIB "-lopenblas -lpthread")
if(USE_MKL_BLAS)
  link_directories(${INTEL_LIBS_DIR})
  message(STATUS "ICC Libraries for MKL: ${INTEL_LIBS_DIR}")
  SET(BLAS_INC_DIR ${MKL_ROOT}/include)
  SET(BLAS_LIB_DIR ${MKL_ROOT}/lib/intel64)
  set(BLAS_LIB "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")
  #set(BLAS_LIB "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")
endif()

# blas library
include_directories(${BLAS_INC_DIR})
link_directories(${BLAS_LIB_DIR})

set(sources
  src/GNNGraph.cpp
  src/GNNLayer.cpp
  src/GNNMath.cpp
  src/GraphConvolutionalLayer.cpp
)

add_library(galois_gnn STATIC ${sources})
target_link_libraries(galois_gnn galois_shmem)
target_link_libraries(galois_gnn ${MPI_CXX_LIBRARIES})
target_link_libraries(galois_gnn ${BLAS_LIB} ${BOOST_LIBRARIES})
target_link_libraries(galois_gnn galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(galois_gnn PROPERTIES EXPORT_NAME gluon)

add_subdirectory(test)
