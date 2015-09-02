include_directories(${CUDA_INCLUDE_DIRS} )
cuda_add_library(pr_cuda pr_cuda.cu)
target_link_libraries(pr_cuda ggrt)
app(pr pr.cpp DISTSAFE)
target_link_libraries(pr pr_cuda cudadevrt OpenCL)
add_custom_command(TARGET pr POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/*.cl $<TARGET_FILE_DIR:pr>)


