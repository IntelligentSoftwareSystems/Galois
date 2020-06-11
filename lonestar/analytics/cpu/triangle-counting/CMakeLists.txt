add_executable(triangle-counting-cpu Triangles.cpp)
add_dependencies(apps triangle-counting-cpu)
target_link_libraries(triangle-counting-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS triangle-counting-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small-node triangle-counting-cpu -symmetricGraph -algo=nodeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
add_test_scale(small-edge triangle-counting-cpu -symmetricGraph -algo=edgeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
