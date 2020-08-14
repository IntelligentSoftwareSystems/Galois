add_library(miningbench STATIC src/lonestarmine.cpp)
target_include_directories(miningbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(miningbench Galois::shmem LLVMSupport)

if(GALOIS_ENABLE_GPU)
  add_library(miningbench_gpu STATIC src/lonestarmine.cpp)
  target_include_directories(miningbench_gpu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_definitions(miningbench_gpu PRIVATE GALOIS_ENABLE_GPU=1) 
  target_link_libraries(miningbench_gpu Galois::shmem LLVMSupport)
endif()
