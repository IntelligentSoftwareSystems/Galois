app(graph-convert graph-convert.cpp)

file(GLOB Sources
  ${CMAKE_SOURCE_DIR}/src/ll/*.cpp
  ${CMAKE_SOURCE_DIR}/src/llvm/*.cpp
  ${CMAKE_SOURCE_DIR}/src/mm/NumaMem.cpp
  )

add_executable(graph-convert-standalone graph-convert.cpp
  ${CMAKE_SOURCE_DIR}/src/FileGraph.cpp ${Sources}
  )
