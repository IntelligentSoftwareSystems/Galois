add_executable(sproute-cpu main.cpp dist.c dl.c err.c heap.c mst2.c neighbors.c bookshelf_IO.c memAlloc.c)
add_dependencies(apps sproute-cpu)
target_link_libraries(sproute-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS sproute-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small1 sproute-cpu -ISPD2008Graph "${BASEINPUT}/eda/routing/test.gr" --flute "${BASEINPUT}/eda/routing")
