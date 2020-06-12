add_executable(sweeps-cpu sweeps.cpp)
add_dependencies(apps sweeps-cpu)
target_link_libraries(sweeps-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS sweeps-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small sweeps-cpu)
