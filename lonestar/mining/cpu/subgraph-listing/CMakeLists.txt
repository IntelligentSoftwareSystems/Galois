add_executable(subgraph-listing-cpu sgl.cpp)
add_dependencies(apps subgraph-listing-cpu)
target_link_libraries(subgraph-listing-cpu PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS subgraph-listing-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
