add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS})

app(hpr_wl HPageRank.cpp)
target_link_libraries(hpr_wl hpr_wl_cuda cudadevrt OpenCL)
add_custom_command(TARGET hpr_wl POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/pagerank_kernel.cl $<TARGET_FILE_DIR:hpr_wl>)
