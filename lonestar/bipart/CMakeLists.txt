add_executable(bipart bipart.cpp Coarsening.cpp Metric.cpp Partitioning.cpp Refine.cpp)
add_dependencies(apps bipart)
target_link_libraries(bipart PRIVATE Galois::shmem lonestar)
install(TARGETS bipart DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
