add_executable(motif motif.cpp)
add_dependencies(apps motif)
target_link_libraries(motif PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS motif DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

#add_test_mine(small1 motif "${BASEINPUT}/Mining/citeseer.csgr")

