add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(bfs-pull bfs_pull.cpp DISTSAFE)
target_link_libraries(bfs-pull hbfs_cuda cudadevrt OpenCL)
add_custom_command(TARGET bfs-pull POST_BUILD COMMAND ${CMAKE_COMMAND}
   -E  copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:bfs-pull>)


