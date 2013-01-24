remove_definitions(-DGALOIS_USE_NUMA)
file(GLOB Sources
  ../graph-convert/graph-convert.cpp
  ${CMAKE_SOURCE_DIR}/src/FileGraph.cpp
  ${CMAKE_SOURCE_DIR}/src/ll/*.cpp
  ${CMAKE_SOURCE_DIR}/src/llvm/*.cpp
  ${CMAKE_SOURCE_DIR}/src/mm/NumaMem.cpp
  )
add_executable(graph-convert-standalone ${Sources})
