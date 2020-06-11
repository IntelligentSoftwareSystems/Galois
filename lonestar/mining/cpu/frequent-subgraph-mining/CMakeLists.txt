add_executable(frequent-subgraph-mining-cpu fsm.cpp)
add_dependencies(apps frequent-subgraph-mining-cpu)
target_link_libraries(frequent-subgraph-mining-cpu PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS frequent-subgraph-mining-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_mine(small1 frequent-subgraph-mining-cpu -symmetricGraph -simpleGraph "${BASEINPUT}/Mining/citeseer.sadj" "-ft=adj" NOT_QUICK)
