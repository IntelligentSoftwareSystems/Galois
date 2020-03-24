include_directories(${OPENCL_INCLUDE_DIRS})
app(cl_test cl_host.cpp)
add_definitions(-D__GALOIS_HET_OPENCL__=1_)
