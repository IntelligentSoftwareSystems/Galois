add_executable(k-core-cpu kcore.cpp)
add_dependencies(apps k-core-cpu)
target_link_libraries(k-core-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS k-core-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small k-core-cpu --kcore=4 -symmetricGraph "${BASEINPUT}/scalefree/symmetric/rmat10.sgr")
