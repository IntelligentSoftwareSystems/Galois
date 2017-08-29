add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
app(cc-pull cc.cpp DISTSAFE)
target_link_libraries(cc-pull cc_cuda cudadevrt OpenCL)
add_custom_command(TARGET cc-pull POST_BUILD COMMAND ${CMAKE_COMMAND} -E
   copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:cc-pull>)


