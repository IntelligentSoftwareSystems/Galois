include_directories(${Eigen_INCLUDE_DIRS})
app(triangles Triangles.cpp REQUIRES Eigen_FOUND)
