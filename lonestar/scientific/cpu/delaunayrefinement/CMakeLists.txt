add_executable(delaunayrefinement-cpu DelaunayRefinement.cpp)
add_dependencies(apps delaunayrefinement-cpu)
target_link_libraries(delaunayrefinement-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS delaunayrefinement-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(delaunayrefinement-cpu PRIVATE -ffast-math)
endif()

add_test_scale(small1 delaunayrefinement-cpu -meshGraph "${BASEINPUT}/reference/meshes/r10k.1")
add_test_scale(small2 delaunayrefinement-cpu -meshGraph "${BASEINPUT}/meshes/250k.2" NOT_QUICK)
