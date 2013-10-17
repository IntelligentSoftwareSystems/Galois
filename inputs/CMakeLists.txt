add_custom_target(more-inputs)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/copyinputs.cmake
  "file(COPY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PATTERN .svn EXCLUDE
  PATTERN CMakeLists.txt EXCLUDE)")
add_custom_target(more-base-graphs COMMAND ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_BINARY_DIR}/copyinputs.cmake)
add_dependencies(more-inputs more-base-graphs)

add_subdirectory(avi)
add_subdirectory(meshes)
add_subdirectory(random)
add_subdirectory(road)
add_subdirectory(scalefree)
