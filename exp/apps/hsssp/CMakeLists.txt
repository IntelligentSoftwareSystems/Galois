#add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS} )
include_directories(${OPENCL_INCLUDE_DIRS})

#app(hsssp HSSSP.cpp DISTSAFE)
#target_link_libraries(hsssp hsssp_cuda cudadevrt OpenCL )
#add_custom_command(TARGET hsssp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/sssp_kernel.cl $<TARGET_FILE_DIR:hsssp>)                 


#app(hsssp-push HSSSP_push.cpp DISTSAFE)
#target_link_libraries(hsssp-push hsssp_cuda cudadevrt OpenCL )
#add_custom_command(TARGET hsssp-push POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/opencl/sssp_kernel.cl $<TARGET_FILE_DIR:hsssp>)                 

#app(graphCheck check.cpp  DISTSAFE)

#app(clCheck CL_graph_test.cpp DISTSAFE)
#target_link_libraries(clCheck OpenCL)

#app(graphCheck_gen check_gen.cpp CLANGPLUGIN)
#app(graphCheck_gen check_gen.cpp  DISTSAFE)

#app(HSSSP_gen HSSSP_gen.cpp  DISTSAFE)
#app(HSSSP_gen2 HSSSP_gen2.cpp  DISTSAFE)

#Working HSSSP
#app(HSSSP_gen3 HSSSP_gen3.cpp DISTSAFE)

#pageRank
#app(PageRank_push pageRank_gen.cpp DISTSAFE)
#app(PageRank_simple pageRank_gen_simpleIter.cpp DISTSAFE)

#LabelPropAlgo
#app(CC_LabelProp connectedComponent_gen.cpp  DISTSAFE)

#sgd
#app(SGD_gen sgd_gen.cpp  DISTSAFE)

#sgd with workList
#app(SGD_gen_wl sgd_wl_gen.cpp  DISTSAFE)

#pageRank_pull
#app(PageRank_pull pageRankPull_gen.cpp  DISTSAFE)
#app(PageRank_pull_new pageRankPull_gen_new.cpp  DISTSAFE)

#HSSSP with worklist
#app(HSSSP_wl HSSSP_wl.cpp  DISTSAFE)


#PageRank with worklist
#app(pageRank_wl pageRank_wl.cpp  DISTSAFE)

#PageRank generated with worklist
#app(pageRank_wl_gen pageRank_wl_gen.cpp  DISTSAFE)

#PageRank with 2 loop structure
#app(pageRank_2loops_gen pageRank_2loops_gen.cpp DISTSAFE)
#OpenCL generated code. 
#app(opencl cl_host.cpp DISTSAFE)
#target_link_libraries(opencl OpenCL)


#PageRank with DGAccum
#app(pageRank_gen_distAccum pageRank_gen_distAccum.cpp DISTSAFE)

#PageRank with vertexCut
#app(pageRank_vertexCut_gen pageRank_vertexCut_gen.cpp DISTSAFE)



#PageRank_test to test compiler
#app(pageRank_gen_test pageRank_gen_test.cpp DISTSAFE)


#PageRank_pull with worklist
#app(pageRankPull_wl pageRankPull_wl.cpp DISTSAFE)

#pagerankPull from stampede
#app(pageRankPull_gen_stampede pageRankPull_gen_stampede.cpp DISTSAFE)


#pagerank vertex cut with 2 loop transform .
#app(pageRank_vertexCut_2loops pageRank_vertexCut_2loops.cpp DISTSAFE)


#pagerank vertex cut with PULL .
#app(pageRankPull_vertexCut pageRankPull_vertexCut.cpp DISTSAFE)

#app(pageRank_wl_gen_test pageRank_wl_gen_test.cpp DISTSAFE)


##################### SSSP ###############################
# SSSP with distAcc, simple push version
#app(sssp_gen sssp_gen.cpp DISTSAFE)


# SSSP with worklist, simple push version
#app(sssp_wl_gen sssp_wl_gen.cpp DISTSAFE)

#SSSP Pull
#app(sssp_pull_gen sssp_pull_gen.cpp DISTSAFE)

#SSSP Pull with worklist
#app(sssp_pull_wl_gen sssp_pull_wl_gen.cpp DISTSAFE)

#SSSP 2 loop transform
#app(sssp_2loop_gen sssp_2loop_gen.cpp DISTSAFE)


#SSSP using vgraph
#app(sssp_vertexCut sssp_vertexCut.cpp DISTSAFE)

##################### BFS ###############################
# BFS with distAcc, simple push version
#app(bfs_gen bfs_gen.cpp DISTSAFE)


# BFS with worklist, simple push version
#app(bfs_wl_gen bfs_wl_gen.cpp DISTSAFE)

#BFS Pull
#app(bfs_pull_gen bfs_pull_gen.cpp DISTSAFE)

#BFS Pull with worklist
#app(bfs_pull_wl_gen bfs_pull_wl_gen.cpp DISTSAFE)

#BFS 2 loop transform
#app(bfs_2loop_gen bfs_2loop_gen.cpp DISTSAFE)

#SSSP using vgraph
#app(bfs_vertexCut bfs_vertexCut.cpp DISTSAFE)

##################### CC ###############################
# CC with distAcc, simple push version
#app(cc_gen cc_gen.cpp DISTSAFE)


# CC with worklist, simple push version
#app(cc_wl_gen cc_wl_gen.cpp DISTSAFE)

#CC Pull
#app(cc_pull_gen cc_pull_gen.cpp DISTSAFE)

#CC Pull with worklist
#app(cc_pull_wl_gen cc_pull_wl_gen.cpp DISTSAFE)

#cc 2 loop transform
#app(cc_2loop_gen cc_2loop_gen.cpp DISTSAFE)

#SSSP using vgraph
#app(cc_vertexCut cc_vertexCut.cpp DISTSAFE)
