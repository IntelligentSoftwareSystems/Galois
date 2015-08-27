if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

app(delaunayrefinement)

add_test_scale(small delaunayrefinement "${BASEINPUT}/meshes/r10k.1")
add_test_scale(web   delaunayrefinement "${BASEINPUT}/meshes/r5M")
