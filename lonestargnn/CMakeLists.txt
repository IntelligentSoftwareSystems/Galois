include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)
include_directories(${CMAKE_SOURCE_DIR}/lonestargnn)
include_directories(${CMAKE_SOURCE_DIR}/libdeepgalois/include)

SET(CUDA_INC /org/centers/cdgc/cuda/cuda-10.0/include)
include_directories(${CUDA_INC})
if(ENABLE_HETERO_GALOIS)
  include_directories(${CMAKE_SOURCE_DIR}/libgpu/include)
endif()

SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS-faraday/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS-faraday/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})
if(NOT ENABLE_HETERO_GALOIS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
endif()

add_subdirectory(gcn)
