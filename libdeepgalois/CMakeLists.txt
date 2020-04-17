cmake_minimum_required(VERSION 2.8)

# open blas
SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS-faraday/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS-faraday/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})

# galois base libs
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libgalois)

if(NOT ENABLE_HETERO_GALOIS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
else()
  # hetero path
  set(CUB_ROOT "${CMAKE_SOURCE_DIR}/cub") # only required headers
  include_directories("${CUB_ROOT}")
  set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/moderngpu") # only required headers
  include_directories("${MGPU_ROOT}/src")

  SET(CUDA_INC /org/centers/cdgc/cuda/cuda-10.0/include)
  include_directories(${CUDA_INC})
  include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)

  find_package(CUDA REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER g++)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_70,code=sm_70)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -G -Xcompiler -rdynamic)
  #set(CUDA_INCLUDE_DIRS /org/centers/cdgc/cuda/cuda-10.0/include ${CUDA_INCLUDE_DIRS})
  SET(CUDA_LIB /org/centers/cdgc/cuda/cuda-10.0/lib64/)
  link_directories(${CUDA_LIB})
  link_directories(${CMAKE_SOURCE_DIR}/libgpu)

  set(CUDA_SOURCES
    src/layers/graph_conv_layer.cu
    src/layers/softmax_loss_layer.cu
    src/layers/aggregator.cu
    src/math_functions.cu
    src/optimizer.cu
    src/context.cu
    src/node.cu
    src/net.cu
  )
  cuda_add_library(dg_gpu ${CUDA_SOURCES})
  target_link_libraries(dg_gpu galois_gpu -lcudart -lcublas -lcusparse -lcurand)
  set_target_properties(dg_gpu PROPERTIES COMPILE_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
  set_target_properties(dg_gpu PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(ENABLE_DIST_GALOIS)
# do not link regular context.cpp; TODO do this conditional in cleaner way
set(sources
  src/layers/graph_conv_layer.cpp
  src/layers/softmax_loss_layer.cpp
  src/layers/aggregator.cpp
  src/layers/layer.cpp
  src/math_functions.cpp
  src/optimizer.cpp
  src/DistContext.cpp
  src/node.cpp
  src/net.cpp
)
else()
set(sources
  src/layers/graph_conv_layer.cpp
  src/layers/softmax_loss_layer.cpp
  src/layers/aggregator.cpp
  src/layers/layer.cpp
  src/math_functions.cpp
  src/optimizer.cpp
  src/context.cpp
  src/sampler.cpp
  src/node.cpp
  src/net.cpp
)
endif()

add_library(dg_cpu STATIC ${sources})
target_link_libraries(dg_cpu galois_shmem gllvm)
target_link_libraries(dg_cpu ${MPI_CXX_LIBRARIES})
target_link_libraries(dg_cpu -lopenblas)
target_include_directories(dg_cpu PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# dist galois setup/linking to dg_cpu
if(ENABLE_DIST_GALOIS)
  target_link_libraries(dg_cpu galois_dist_async galois_cusp galois_gluon)
  target_include_directories(dg_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/libdist/include
    ${CMAKE_SOURCE_DIR}/libcusp/include
    ${CMAKE_SOURCE_DIR}/libgluon/include
  )

  if(ENABLE_HETERO_GALOIS)
    target_link_libraries(dg_gpu galois_dist_async galois_cusp galois_gluon)
    target_include_directories(dg_gpu PUBLIC
      ${CMAKE_SOURCE_DIR}/libdist/include
      ${CMAKE_SOURCE_DIR}/libcusp/include
      ${CMAKE_SOURCE_DIR}/libgluon/include
    )
  endif()
endif()

set_target_properties(dg_cpu PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE On
  POSITION_INDEPENDENT_CODE On
)
