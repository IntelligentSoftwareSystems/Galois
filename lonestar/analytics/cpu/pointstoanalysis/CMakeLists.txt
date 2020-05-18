add_executable(pta PointsTo.cpp)
add_dependencies(apps pta)
target_link_libraries(pta PRIVATE Galois::shmem lonestar)
install(TARGETS pta DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small pta "${BASEINPUT}/java/pta/gap_constraints.txt")
