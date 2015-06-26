add_subdirectory(cuda)
app(hpagerank HPageRank.cpp DISTSAFE)
target_link_libraries(hpagerank hpr_cuda cudadevrt)
