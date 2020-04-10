add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS})
app(cc-push cc_push.cpp)
target_link_libraries(cc-push cc_cuda_push cudadevrt OpenCL)
add_custom_command(TARGET cc-push POST_BUILD COMMAND ${CMAKE_COMMAND} -E
   copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
   $<TARGET_FILE_DIR:cc-push>)


