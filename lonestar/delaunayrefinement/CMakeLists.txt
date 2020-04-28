add_executable(delaunayrefinement DelaunayRefinement.cpp)
add_dependencies(apps delaunayrefinement)
target_link_libraries(delaunayrefinement PRIVATE Galois::shmem lonestar)
install(TARGETS delaunayrefinement DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(delaunayrefinement PRIVATE -ffast-math)
endif()

add_test_scale(small1 delaunayrefinement "${BASEINPUT}/reference/meshes/r10k.1")
add_test_scale(small2 delaunayrefinement "${BASEINPUT}/meshes/250k.2" NOT_QUICK)
