cmake_minimum_required(VERSION 2.8)

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
message(STATUS "BLAS_INC_DIR: ${BLAS_INC_DIR}")
message(STATUS "BLAS_LIB_DIR: ${BLAS_LIB_DIR}")

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
  include_directories(${CUDA_HOME}/include)
  include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)

  find_package(CUDA REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER g++)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_70,code=sm_70)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -G -Xcompiler -rdynamic)
  link_directories(${CUDA_HOME}/lib64)
  link_directories(${CMAKE_SOURCE_DIR}/libgpu)
  message(STATUS "CUDA_LIB_DIR: ${CUDA_HOME}/lib64")

  set(CUDA_SOURCES
    src/layers/graph_conv_layer.cu
    src/layers/softmax_loss_layer.cu
    src/layers/sigmoid_loss_layer.cu
    src/layers/leaky_relu_layer.cu
    src/layers/l2_norm_layer.cu
    src/layers/relu_layer.cu
    src/layers/aggregator.cu
    src/math_functions.cu
    src/optimizer.cu
    src/context.cu
    src/lgraph.cu
    src/node.cu
    src/net.cu
  )
  cuda_add_library(dg_gpu ${CUDA_SOURCES})
  target_link_libraries(dg_gpu galois_gpu -lcudart -lcublas -lcusparse -lcurand)
  set_target_properties(dg_gpu PROPERTIES COMPILE_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
  set_target_properties(dg_gpu PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(ENABLE_HETERO_GALOIS)
set(sources src/net.cpp)
else()
if(ENABLE_DIST_GALOIS)
# do not link regular context.cpp; TODO do this conditional in cleaner way
# also don't link sampler
set(sources
  src/layers/softmax_loss_layer.cpp
  src/layers/sigmoid_loss_layer.cpp
  src/layers/graph_conv_layer.cpp
  src/layers/leaky_relu_layer.cpp
  src/layers/l2_norm_layer.cpp
  src/layers/relu_layer.cpp
  src/layers/aggregator.cpp
  src/math_functions.cpp
  src/layers/layer.cpp
  src/DistContext.cpp
  src/optimizer.cpp
  src/reader.cpp
  src/lgraph.cpp
  src/utils.cpp
  src/node.cpp
  src/net.cpp
)
else()
set(sources
  src/layers/softmax_loss_layer.cpp
  src/layers/sigmoid_loss_layer.cpp
  src/layers/graph_conv_layer.cpp
  src/layers/leaky_relu_layer.cpp
  src/layers/l2_norm_layer.cpp
  src/layers/relu_layer.cpp
  src/layers/aggregator.cpp
  src/math_functions.cpp
  src/layers/layer.cpp
  src/optimizer.cpp
  src/context.cpp
  src/sampler.cpp
  src/reader.cpp
  src/lgraph.cpp
  src/utils.cpp
  src/node.cpp
  src/net.cpp
)
endif()
endif()

#set(BOOST_LIBRARIES "-lboost_system -lboost_thread")
add_library(dg_cpu STATIC ${sources})
target_link_libraries(dg_cpu galois_shmem)
target_link_libraries(dg_cpu ${MPI_CXX_LIBRARIES})
target_link_libraries(dg_cpu ${BLAS_LIB} ${BOOST_LIBRARIES})
target_include_directories(dg_cpu PUBLIC
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
