add_executable(minimum-spanningtree-cpu Boruvka.cpp)
add_dependencies(apps minimum-spanningtree-cpu)
target_link_libraries(minimum-spanningtree-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS minimum-spanningtree-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small1 minimum-spanningtree-cpu "${BASEINPUT}/scalefree/rmat10.gr")
add_test_scale(small2 minimum-spanningtree-cpu "${BASEINPUT}/reference/structured/rome99.gr")
