set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
app(delaunaytriangulation DelaunayTriangulation.cpp Element.cpp)
app(dt2 DelaunayTriangulation2.cpp Element.cpp)
