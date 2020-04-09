add_executable(bip bipart.cpp Coarsening.cpp Metric.cpp Partitioning.cpp Refine.cpp)
add_dependencies(apps bip)
target_link_libraries(bip PRIVATE Galois::shmem lonestar)
install(TARGETS bip DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
