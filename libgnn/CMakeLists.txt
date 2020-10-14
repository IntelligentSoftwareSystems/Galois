set(BLAS_INC_DIR ${OPENBLAS_ROOT}/include/openblas)
set(BLAS_LIB_DIR ${OPENBLAS_ROOT}/lib64)
# blas library
include_directories(${BLAS_INC_DIR})
link_directories(${BLAS_LIB_DIR})

set(BLAS_LIB "-lopenblas")

set(sources
  src/GNNGraph.cpp
  src/GNNLayer.cpp
  src/GNNMath.cpp
  src/GraphConvolutionalLayer.cpp
  src/SoftmaxLayer.cpp
  src/GraphNeuralNetwork.cpp
  src/GNNOptimizers.cpp
  src/GluonGradientInterface.cpp
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
