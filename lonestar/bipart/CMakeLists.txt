set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
app(bipart main.cpp Coarsening.cpp Refine.cpp Partitioning.cpp Metric.cpp)

