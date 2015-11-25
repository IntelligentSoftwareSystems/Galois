add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(hbfs-pull bfs_pull.cpp DISTSAFE)
target_link_libraries(hbfs-pull hbfs_cuda cudadevrt OpenCL)
#target_link_libraries(sssp_gen cuda OpenCL)
add_custom_command(TARGET hbfs-pull POST_BUILD COMMAND ${CMAKE_COMMAND}
   -E  copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:hbfs-pull>)


