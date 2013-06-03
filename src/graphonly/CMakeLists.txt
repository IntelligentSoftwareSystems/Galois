add_definitions(-DGALOIS_FORCE_NO_NUMA)
file(GLOB Sources
  ${CMAKE_SOURCE_DIR}/src/FileGraph.cpp
  ${CMAKE_SOURCE_DIR}/src/ll/*.cpp
  ${CMAKE_SOURCE_DIR}/src/llvm/*.cpp
  ${CMAKE_SOURCE_DIR}/src/mm/*.cpp
  )
add_library(galois-graph ${Sources})
if(GALOIS_ADDITIONAL_SYSTEM_LIBRARIES)
  target_link_libraries(galois-graph ${GALOIS_ADDITIONAL_SYSTEM_LIBRARIES})
endif()
