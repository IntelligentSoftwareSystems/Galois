add_executable(triangle-counting-mining-cpu tc_mine.cpp)
add_dependencies(apps triangle-counting-mining-cpu)
target_link_libraries(triangle-counting-mining-cpu PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS triangle-counting-mining-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_mine(small1 triangle-counting-mining-cpu -symmetricGraph -simpleGraph "${BASEINPUT}/Mining/citeseer.csgr" NOT_QUICK)
