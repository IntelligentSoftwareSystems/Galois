add_executable(motif-counting-cpu motif.cpp)
add_dependencies(apps motif-counting-cpu)
target_link_libraries(motif-counting-cpu PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS motif-counting-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_mine(small1 motif-counting-cpu -symmetricGraph -simpleGraph "${BASEINPUT}/Mining/citeseer.csgr" NOT_QUICK)
