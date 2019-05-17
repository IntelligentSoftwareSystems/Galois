if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

app(delaunaytriangulation DelaunayTriangulation.cpp Element.cpp)
app(delaunaytriangulation-det DelaunayTriangulationDet.cpp Element.cpp)

add_test_scale(small1 delaunaytriangulation "${BASEINPUT}/reference/meshes/r10k.node")
add_test_scale(small2 delaunaytriangulation "${BASEINPUT}/meshes/250k.2.node")
#add_test_scale(web delaunaytriangulation "${BASEINPUT}/meshes/r5M.node")

add_test_scale(small1 delaunaytriangulation-det "${BASEINPUT}/reference/meshes/r10k.node")
add_test_scale(small2 delaunaytriangulation-det "${BASEINPUT}/meshes/250k.2.node")
#add_test_scale(web delaunaytriangulation-det "${BASEINPUT}/meshes/r5M.node")
