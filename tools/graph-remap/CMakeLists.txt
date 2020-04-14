add_executable(graph-remap graph-remap.cpp)
target_link_libraries(graph-remap PRIVATE galois_shmem LLVMSupport)
