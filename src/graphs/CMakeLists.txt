project(gGraphs)

set(sources
        src/FileGraph.cpp
        src/FileGraphParallel_cpp11.cpp
#        src/FileGraphParallel_pthread.cpp
        src/OCFileGraph.cpp
)

add_library(galois_graphs STATIC ${sources})

#target_include_directories(galois_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
