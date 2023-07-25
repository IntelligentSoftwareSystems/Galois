set(sources
  src/DistributedMinibatchTracker.cpp
  src/GNNMath.cpp
  src/GNNOptimizers.cpp
  src/GraphNeuralNetwork.cpp
  src/MinibatchGenerator.cpp
  src/graphs/GNNGraph.cpp
  src/graphs/GNNSubgraph.cpp
  src/layers/DenseLayer.cpp
  src/layers/GNNLayer.cpp
  src/layers/GraphConvolutionalLayer.cpp
  src/layers/L2NormLayer.cpp
  src/layers/SAGELayer.cpp
  src/layers/SigmoidLayer.cpp
  src/layers/SoftmaxLayer.cpp
)

## TODO(hc): Note that these libraries should be hard-coded
## based on your own system.
## These should be automatic library linking.
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64)
set(INTEL_COMPILER_LIBRARIES /home/hochan/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin)
set(INTEL_LIBS "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5")
set(SINGLE_INTEL_LIBS "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")

add_library(galois_gnn STATIC ${sources})

target_link_directories(galois_gnn PUBLIC ${MKL_LIBRARIES})
target_link_directories(galois_gnn PUBLIC ${INTEL_COMPILER_LIBRARIES})
target_link_libraries(galois_gnn ${INTEL_LIBS})
target_link_libraries(galois_gnn galois_shmem)
target_link_libraries(galois_gnn galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MKL_INCLUDE_DIRS}
)

add_library(galois_gnn_single STATIC ${sources})
target_link_directories(galois_gnn_single PUBLIC ${MKL_LIBRARIES})
target_link_libraries(galois_gnn_single galois_shmem)
target_link_libraries(galois_gnn_single ${SINGLE_INTEL_LIBS})
target_link_libraries(galois_gnn_single galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn_single PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MKL_INCLUDE_DIRS}
)

set_target_properties(galois_gnn PROPERTIES EXPORT_NAME galois_gnn)
add_subdirectory(test)

if (GALOIS_ENABLE_GPU)
  target_compile_definitions(galois_gnn PUBLIC GALOIS_ENABLE_GPU=1)

  # create the galois_gnn_gpu library to get linked into galois_gnn
  set(gpusources
    src/GNNMath.cu
    src/graphs/GNNGraph.cu
    src/layers/GNNLayer.cu
    src/layers/GraphConvolutionalLayer.cu
    src/layers/SoftmaxLayer.cu
    src/layers/SAGELayer.cu
    src/GraphNeuralNetwork.cu
    src/GNNOptimizers.cu
    src/GNNCudaContext.cu
    src/CUDAUtil.cu
  )
  add_library(galois_gnn_gpu STATIC ${gpusources})
  target_compile_definitions(galois_gnn_gpu PRIVATE _FORCE_INLINES)
  target_compile_options(galois_gnn_gpu PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
  set_property(TARGET galois_gnn_gpu PROPERTY CUDA_STANDARD 14)
  target_compile_definitions(galois_gnn_gpu PUBLIC GALOIS_ENABLE_GPU=1)
  target_include_directories(galois_gnn_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # link to gpu lib (which takes care of moderngpu and cub) as well as cu libs
  target_link_libraries(galois_gnn_gpu galois_gluon Galois::gpu galois_support -lcublas -lcurand)

  # gpu -> cpu lib
  target_link_libraries(galois_gnn galois_gnn_gpu)
endif()
