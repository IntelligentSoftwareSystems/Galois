add_executable(pagerank-pull-cpu PageRank-pull.cpp)
add_dependencies(apps pagerank-pull-cpu)
target_link_libraries(pagerank-pull-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS pagerank-pull-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small pagerank-pull-cpu -transposedGraph -tolerance=0.01 "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
add_test_scale(small-topo pagerank-pull-cpu -transposedGraph -tolerance=0.01 -algo=Topo "${BASEINPUT}/scalefree/transpose/rmat10.tgr")

add_executable(pagerank-push-cpu PageRank-push.cpp)
add_dependencies(apps pagerank-push-cpu)
target_link_libraries(pagerank-push-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS pagerank-push-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small pagerank-push-cpu -tolerance=0.01 "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
add_test_scale(small-sync pagerank-push-cpu -tolerance=0.01 -algo=Sync "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
