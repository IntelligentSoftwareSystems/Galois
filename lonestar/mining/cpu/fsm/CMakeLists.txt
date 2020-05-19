add_executable(fsm fsm.cpp)
add_dependencies(apps fsm)
target_link_libraries(fsm PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS fsm DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

#add_test_mine(small1 fsm "${BASEINPUT}/Mining/citeseer.sadj -ft=adj")

