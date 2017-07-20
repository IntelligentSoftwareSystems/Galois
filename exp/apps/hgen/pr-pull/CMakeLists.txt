add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
#cuda_add_library(pr_cuda pr_cuda.cu)
#target_link_libraries(pr_cuda ggrt)
app(pr-pull pr.cpp DISTSAFE)
target_link_libraries(pr-pull pr_cuda cudadevrt OpenCL)
add_custom_command(TARGET pr-pull POST_BUILD COMMAND ${CMAKE_COMMAND} -E
   copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:pr-pull>)


