if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

app(delaunaytriangulation DelaunayTriangulation.cpp Element.cpp)
app(delaunaytriangulation-det DelaunayTriangulationDet.cpp Element.cpp)

add_test_scale(small delaunaytriangulation "${BASEINPUT}/meshes/r10k.node")
add_test_scale(web   delaunaytriangulation "${BASEINPUT}/meshes/r5M.node")
