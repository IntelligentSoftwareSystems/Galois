#add_subdirectory(cuda)
#include_directories(${CUDA_INCLUDE_DIRS} )

app(hpagerank HPageRank.cpp)
target_link_libraries(hpagerank hpr_cuda cudadevrt OpenCL)
add_custom_command(TARGET hpagerank POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/pagerank_kernel.cl $<TARGET_FILE_DIR:hpagerank>)

app(hpagerank-term HPageRankTerm.cpp)
target_link_libraries(hpagerank-term hpr_cuda cudadevrt OpenCL)
add_custom_command(TARGET hpagerank-term POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/pagerank_kernel.cl $<TARGET_FILE_DIR:hpagerank>)

app(hpagerank_push HPageRank_push.cpp)
target_link_libraries(hpagerank_push hpr_cuda cudadevrt OpenCL)
add_custom_command(TARGET hpagerank_push POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/pagerank_kernel.cl $<TARGET_FILE_DIR:hpagerank_push>)

app(hpagerank_push_serialReduce HPageRank_push_serialReduce.cpp)
target_link_libraries(hpagerank_push_serialReduce hpr_cuda cudadevrt OpenCL)
add_custom_command(TARGET hpagerank_push_serialReduce POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/pagerank_kernel.cl $<TARGET_FILE_DIR:hpagerank_push_serialReduce>)

#without CUDA and OpenCL
app(hpagerank_push_NoOpenCL HPageRank_push_withoutOpenCL.cpp)
