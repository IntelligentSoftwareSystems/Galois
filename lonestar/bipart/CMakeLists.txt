add_executable(bipart main.cpp Coarsening.cpp Refine.cpp Partitioning.cpp Metric.cpp)
add_dependencies(apps bipart)
target_link_libraries(bipart PRIVATE galois_shmem lonestar)
install(TARGETS bipart DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

