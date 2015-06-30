add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(hpagerank HPageRank.cpp DISTSAFE)
target_link_libraries(hpagerank hpr_cuda cudadevrt OpenCL )
