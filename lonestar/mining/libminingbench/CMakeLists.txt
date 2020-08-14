add_library(miningbench STATIC src/Start.cpp src/Input.cpp)
target_include_directories(miningbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(miningbench Galois::shmem LLVMSupport lonestar)

if(GALOIS_ENABLE_DIST)
  add_library(miningbench_dist STATIC src/Start.cpp)
  target_include_directories(miningbench_dist PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_definitions(miningbench_dist PRIVATE GALOIS_ENABLE_DIST=1) 
  target_link_libraries(miningbench_dist Galois::shmem LLVMSupport distbench)
endif()

if(GALOIS_ENABLE_GPU)
  add_library(miningbench_gpu STATIC src/Start.cpp src/Input.cpp)
  target_include_directories(miningbench_gpu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_definitions(miningbench_gpu PRIVATE GALOIS_ENABLE_GPU=1) 
  target_link_libraries(miningbench_gpu Galois::shmem LLVMSupport)
endif()
