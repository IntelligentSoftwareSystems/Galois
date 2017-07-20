include_directories(${OPENCL_INCLUDE_DIRS})
#app(cl_test_src  bfs_pull-topological.cpp DISTSAFE)
app(cl_test cl_host.cpp DISTSAFE)
add_definitions(-D__GALOIS_HET_OPENCL__=1_)

#Nvidia - opencl
#target_link_libraries(cl_test OpenCL )

#FPGA 
#add_definitions(-D_ALTERA_FPGA_USE_=1_)
#set(ALTERA_LIB_PATH $ENV{ALTERAOCLSDKROOT}/linux64/lib)
#set(ALTERA_HOST_LIB_PATH $ENV{ALTERAOCLSDKROOT}/host/linux64/lib)
#target_link_libraries(cl_test ${ALTERA_LIB_PATH}/libalteracl.so
#   ${ALTERA_HOST_LIB_PATH}/libacl_emulator_kernel_rt.so 
#   ${ALTERA_LIB_PATH}/libalterahalmmd.so ${ALTERA_LIB_PATH}/libelf.so
#   ${ALTERA_LIB_PATH}/libOpenCL.so  )

