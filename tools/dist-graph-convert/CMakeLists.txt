app(dist-graph-convert dist-graph-convert.cpp)
target_link_libraries(dist-graph-convert galois_dist)
target_include_directories(dist-graph-convert PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
