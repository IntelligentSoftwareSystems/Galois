add_executable(gmetis Coarsening.cpp GMetis.cpp Metric.cpp Partitioning.cpp Refine.cpp)
add_dependencies(apps gmetis)
target_link_libraries(gmetis PRIVATE Galois::shmem lonestar)
install(TARGETS gmetis DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
# Disable failing test (issue #116).
add_test_scale(small1 gmetis "${BASEINPUT}/reference/structured/rome99.gr" 4 NOT_QUICK)
add_test_scale(small2 gmetis "${BASEINPUT}/scalefree/rmat10.gr" 256)
