include_directories(${CUDA_INCLUDE_DIRS} )
cuda_add_library(pr_cuda pr_cuda.cu)
target_link_libraries(pr_cuda ggrt)
app(pr pr.cpp DISTSAFE)
target_link_libraries(pr pr_cuda cudadevrt )

