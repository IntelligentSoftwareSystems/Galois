remove_definitions(-DGALOIS_USE_NUMA)
file(GLOB Sources
  ${CMAKE_SOURCE_DIR}/src/FileGraph.cpp
  ${CMAKE_SOURCE_DIR}/src/ll/*.cpp
  ${CMAKE_SOURCE_DIR}/src/llvm/*.cpp
  ${CMAKE_SOURCE_DIR}/src/mm/NumaMem.cpp
  )
add_library(standalone ${Sources})
add_executable(graph-convert-standalone ../graph-convert/graph-convert.cpp)
target_link_libraries(graph-convert-standalone standalone)
