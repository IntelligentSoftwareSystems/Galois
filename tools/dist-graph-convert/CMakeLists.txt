add_executable(dist-graph-convert dist-graph-convert.cpp dist-graph-convert-helpers.cpp)

target_link_libraries(dist-graph-convert PRIVATE galois_dist_async LLVMSupport)
