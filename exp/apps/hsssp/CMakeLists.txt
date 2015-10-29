add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )

app(hsssp HSSSP.cpp DISTSAFE)
target_link_libraries(hsssp hsssp_cuda cudadevrt OpenCL )
add_custom_command(TARGET hsssp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/sssp_kernel.cl $<TARGET_FILE_DIR:hsssp>)                 


app(hsssp-push HSSSP_push.cpp DISTSAFE)
target_link_libraries(hsssp-push hsssp_cuda cudadevrt OpenCL )
add_custom_command(TARGET hsssp-push POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/sssp_kernel.cl $<TARGET_FILE_DIR:hsssp>)                 

app(graphCheck check.cpp GlobalObj.cpp DISTSAFE)

#Working HSSSP
app(HSSSP_gen3 HSSSP_gen3.cpp GlobalObj.cpp DISTSAFE)

#pageRank
app(PageRank pageRank_gen.cpp GlobalObj.cpp DISTSAFE)
