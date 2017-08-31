add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(bfs-push bfs_push.cpp DISTSAFE)
target_link_libraries(bfs-push hbfs_push_cuda cudadevrt OpenCL)
#target_link_libraries(sssp_gen cuda OpenCL)
add_custom_command(TARGET bfs-push POST_BUILD COMMAND ${CMAKE_COMMAND}
   -E  copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:bfs-push>)


