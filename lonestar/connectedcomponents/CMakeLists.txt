add_executable(connectedcomponents ConnectedComponents.cpp)
add_dependencies(apps connectedcomponents)
target_link_libraries(connectedcomponents PRIVATE galois_shmem lonestar)
install(TARGETS connectedcomponents DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small connectedcomponents "${BASEINPUT}/scalefree/symmetric/rmat10.sgr")
