set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

find_package(Eigen)
if(Eigen_FOUND)
  include_directories(${Eigen_INCLUDE_DIRS})
  add_definitions(-DHAS_EIGEN -DEIGEN_DONT_PARALLELIZE)
endif()

app(mc mc.cpp)
