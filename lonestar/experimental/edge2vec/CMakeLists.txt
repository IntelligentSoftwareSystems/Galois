add_executable(edge2vec edge2vec.cpp)
add_dependencies(apps edge2vec)
target_link_libraries(edge2vec PRIVATE Galois::shmem lonestar)
install(TARGETS edge2vec DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT apps EXCLUDE_FROM_ALL)

#add_te
#add_test_scale(small2 bfs "${BASEINPUT}/scalefree/rmat10.gr")
#add_test_scale(web bfs "${BASEINPUT}/random/r4-2e26.gr")
