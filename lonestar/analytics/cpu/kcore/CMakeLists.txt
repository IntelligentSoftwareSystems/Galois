add_executable(kcore kcore.cpp)
add_dependencies(apps kcore)
target_link_libraries(kcore PRIVATE Galois::shmem lonestar)
install(TARGETS kcore DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small kcore --kcore=4 -symmetricGraph "${BASEINPUT}/scalefree/symmetric/rmat10.sgr")
