#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
#set(CMAKE_VERBOSE_MAKEFILE on )

cmake_minimum_required(VERSION 2.8)
include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)
if(USE_GPU)
  set(CUB_ROOT "${CMAKE_SOURCE_DIR}/cub") # only required headers
  set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/moderngpu") # only required headers
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_GPU")
  include_directories(${CMAKE_SOURCE_DIR}/libpangolin/gpu_mining)
else()
  include_directories(${CMAKE_SOURCE_DIR}/libpangolin/include)
endif()

function(mineApp name)
  if(USE_GPU)
    find_package(CUDA REQUIRED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_HOST_COMPILER g++)
    include_directories("${CUB_ROOT}")
    include_directories("${MGPU_ROOT}/src")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_70,code=sm_70)
    #target_link_libraries(${name} ${name}_cuda)
    FILE(GLOB CUSOURCES ${name}*.cu)
    cuda_add_library(${name}_cuda ${CUSOURCES})
    app(${name}_gpu ${name}_gpu.cpp)
    target_link_libraries(${name}_gpu ${name}_cuda)
  else()
    app(${name} ${name}.cpp)
    target_link_libraries(${name} pangolin)
  endif()
endfunction()

add_subdirectory(tc)
add_subdirectory(kcl)
add_subdirectory(motif)
add_subdirectory(fsm)
#add_subdirectory(sgl)

