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

set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64)
set(INTEL_LIBS "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")

add_library(galois_gnn STATIC ${sources})
target_link_directories(galois_gnn PUBLIC ${MKL_LIBRARIES})
target_link_libraries(galois_gnn galois_shmem)
target_link_libraries(galois_gnn ${INTEL_LIBS})
target_link_libraries(galois_gnn galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MKL_INCLUDE_DIRS}
)

set_target_properties(galois_gnn PROPERTIES EXPORT_NAME galois_gnn)

add_subdirectory(test)
