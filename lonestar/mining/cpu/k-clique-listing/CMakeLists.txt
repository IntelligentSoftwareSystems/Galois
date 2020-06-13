add_executable(k-clique-listing-cpu kcl.cpp)
add_dependencies(apps k-clique-listing-cpu)
target_link_libraries(k-clique-listing-cpu PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS k-clique-listing-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_mine(small1 k-clique-listing-cpu -symmetricGraph -simpleGraph "${BASEINPUT}/Mining/citeseer.csgr" NOT_QUICK)
