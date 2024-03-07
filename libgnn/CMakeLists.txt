set(sources
  src/DistributedMinibatchTracker.cpp
  src/GNNMath.cpp
  src/GNNOptimizers.cpp
  src/MinibatchGenerator.cpp
  src/graphs/GNNGraph.cpp
)

add_library(galois_gnn STATIC ${sources})

target_compile_options(galois_gnn PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(galois_gnn PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(galois_gnn PUBLIC $<LINK_ONLY:MKL::MKL>)
target_link_libraries(galois_gnn PUBLIC galois_shmem)
target_link_libraries(galois_gnn PUBLIC galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(galois_gnn_single STATIC ${sources})
target_compile_options(galois_gnn_single PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(galois_gnn_single PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(galois_gnn_single PUBLIC $<LINK_ONLY:MKL::MKL>)
target_link_libraries(galois_gnn_single PUBLIC galois_shmem)
target_link_libraries(galois_gnn_single PUBLIC galois_dist_async galois_cusp galois_gluon galois_support)
target_include_directories(galois_gnn_single PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
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
