add_executable(maximum-cardinality-matching-cpu bipartite-mcm.cpp)
add_dependencies(apps maximum-cardinality-matching-cpu)
target_link_libraries(maximum-cardinality-matching-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS maximum-cardinality-matching-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small1 maximum-cardinality-matching-cpu -symmetricGraph -inputType generated -n 100 -numEdges 1000 -numGroups 10 -seed 0)
add_test_scale(small2 maximum-cardinality-matching-cpu -symmetricGraph -inputType generated -n 100 -numEdges 10000 -numGroups 100 -seed 0)
