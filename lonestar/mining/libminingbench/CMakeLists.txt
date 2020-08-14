add_library(miningbench STATIC src/lonestarmine.cpp)
target_include_directories(miningbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(miningbench Galois::shmem LLVMSupport)
