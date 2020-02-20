SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})

#SET(CUDA_INC /org/centers/cdgc/cuda/cuda-10.0/include)
#SET(CUDA_LIB /org/centers/cdgc/cuda/cuda-10.0/lib64/)
#SET(ENABLE_GPU OFF CACHE BOOL "Use GPU for DeepGalois")
#if (ENABLE_GPU)
#	target_compile_definitions(distbench PRIVATE __GALOIS_HET_CUDA__=1)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_GPU")
#    include_directories(${CUDA_INC})
#    link_directories(${CUDA_LIB})
#endif()

#set(sources
#  src/layers/relu_layer.cu
#)
#cuda_add_library(deepgalois_gpu
#  ${sources}
#OPTIONS -D_FORCE_INLINES
#)
#target_include_directories(deepgalois_gpu PUBLIC
#  ${CMAKE_SOURCE_DIR}/libgpu/include
#)
#set_target_properties(deepgalois_gpu PROPERTIES
#  INTERFACE_POSITION_INDEPENDENT_CODE On
#  POSITION_INDEPENDENT_CODE On
#)
#target_link_libraries(deepgalois -lcudart -lcublas)

set(sources
  src/layers/relu_layer.cpp
  src/layers/softmax_loss_layer.cpp
  src/math_functions.cpp
)
add_library(deepgalois STATIC ${sources})

target_link_libraries(deepgalois galois_shmem gllvm)
target_link_libraries(deepgalois ${MPI_CXX_LIBRARIES})
target_link_libraries(deepgalois -lopenblas)

target_include_directories(deepgalois PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(deepgalois PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
