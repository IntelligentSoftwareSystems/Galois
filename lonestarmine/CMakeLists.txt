include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)
include_directories(${CMAKE_SOURCE_DIR}/libpangolin)
if(USE_GPU)
  include_directories(${CMAKE_SOURCE_DIR}/libpangolin/gpu_mining)
endif()

function(mineApp name)
  if(USE_GPU)
    target_link_libraries(${name} ${name}_cuda)
    FILE(GLOB CUSOURCES ${name}*.cu)
    cuda_add_library(${name}_cuda ${CUSOURCES})
    #target_link_libraries(${name}_cuda galois_gpu)
  endif()
endfunction()

add_subdirectory(tc)
add_subdirectory(kcl)
add_subdirectory(motif)
add_subdirectory(fsm)
add_subdirectory(sgl)

