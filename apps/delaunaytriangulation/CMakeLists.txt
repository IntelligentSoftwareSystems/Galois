set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
app(delaunaytriangulation DelaunayTriangulation.cpp Element.cpp)
app(DelaunayTriangulation-det DelaunayTriangulationDet.cpp Element.cpp)
