add_executable(kcl kcl.cpp)
add_dependencies(apps kcl)
target_link_libraries(kcl PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS kcl DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

#add_test_mine(small1 kcl "${BASEINPUT}/Mining/citeseer.csgr")

