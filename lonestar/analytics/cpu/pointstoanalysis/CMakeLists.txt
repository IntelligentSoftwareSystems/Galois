add_executable(pointstoanalysis-cpu PointsTo.cpp)
add_dependencies(apps pointstoanalysis-cpu)
target_link_libraries(pointstoanalysis-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS pointstoanalysis-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small pointstoanalysis-cpu "${BASEINPUT}/java/pta/gap_constraints.txt")
