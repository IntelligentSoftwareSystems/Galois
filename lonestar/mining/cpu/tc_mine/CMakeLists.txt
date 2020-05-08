add_executable(tc_mine tc_mine.cpp)
add_dependencies(apps tc_mine)
target_link_libraries(tc_mine PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS tc_mine DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

#add_test_mine(small1 tc_mine "${BASEINPUT}/Mining/citeseer.csgr")

