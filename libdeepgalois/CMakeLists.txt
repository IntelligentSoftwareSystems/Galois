SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})

SET(CUDA_INC /org/centers/cdgc/cuda/cuda-8.0/include)
SET(CUDA_LIB /org/centers/cdgc/cuda/cuda-8.0/lib64/)
include_directories(${CUDA_INC})
link_directories(${CUDA_LIB})
link_directories(${CMAKE_SOURCE_DIR}/libgpu)

#set(sources
#  src/layers/relu_layer.cu
#)
#target_include_directories(deepgalois_gpu PUBLIC
#  ${CMAKE_SOURCE_DIR}/libgpu/include
#)
#set_target_properties(deepgalois_gpu PROPERTIES
#  INTERFACE_POSITION_INDEPENDENT_CODE On
#  POSITION_INDEPENDENT_CODE On
#)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60; -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)
file(GLOB CUDA_FILES "src/" *.cu)
#CUDA_COMPILE(CU_O src/math_functions.cu)
CUDA_COMPILE(CU_O ${CUDA_FILES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(sources
  src/layers/relu_layer.cpp
  src/layers/graph_conv_layer.cpp
  src/layers/softmax_loss_layer.cpp
  src/math_functions.cpp
  src/aggregator.cpp
  ${CU_O}
)
add_library(deepgalois STATIC ${sources})

target_link_libraries(deepgalois galois_shmem gllvm galois_gpu)
target_link_libraries(deepgalois ${MPI_CXX_LIBRARIES})
target_link_libraries(deepgalois -lopenblas)
target_link_libraries(deepgalois -lcudart -lcublas)

target_include_directories(deepgalois PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(deepgalois PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
