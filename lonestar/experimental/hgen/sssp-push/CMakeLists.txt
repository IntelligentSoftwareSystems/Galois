add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(sssp-push sssp_push.cpp DISTSAFE)
target_link_libraries(sssp-push sssp_cuda_push cudadevrt OpenCL)
#target_link_libraries(sssp_gen cuda OpenCL)
add_custom_command(TARGET sssp-push POST_BUILD COMMAND
   ${CMAKE_COMMAND} -E  copy_if_different
   ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:sssp-push>)


