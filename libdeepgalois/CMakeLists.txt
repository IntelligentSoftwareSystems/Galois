cmake_minimum_required(VERSION 2.8)

SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)
set(CUB_ROOT "${CMAKE_SOURCE_DIR}/cub") # only required headers
set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/moderngpu") # only required headers
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})
link_directories(${CMAKE_SOURCE_DIR}/libgalois)

#deepgalois_option(CPU_ONLY "Build DeepGalois without CUDA support" OFF)
set(CPU_ONLY ON CACHE BOOL "Build DeepGalois without CUDA support")
if(CPU_ONLY)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
else()
  SET(CUDA_INC /org/centers/cdgc/cuda/cuda-8.0/include)
  SET(CUDA_LIB /org/centers/cdgc/cuda/cuda-8.0/lib64/)
  include_directories(${CUDA_INC})
  link_directories(${CUDA_LIB})
  link_directories(${CMAKE_SOURCE_DIR}/libgpu)
  find_package(CUDA REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER g++)
  #list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")
  #list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60; -std=c++11")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)
  include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)
  #include_directories(${CUB_ROOT})
  #include_directories(${MGPU_ROOT}/src)
  cuda_include_directories("${CUB_ROOT}")
  cuda_include_directories("${MGPU_ROOT}/src")
  file(GLOB CUDA_FILES "src/" *.cu)
  CUDA_COMPILE(CU_O src/math_functions.cu src/aggregator.cu)
  #CUDA_COMPILE(CU_O ${CUDA_FILES})
endif()

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
