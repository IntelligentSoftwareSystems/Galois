file(COPY . DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PATTERN .svn EXCLUDE PATTERN CMakeLists.txt EXCLUDE)

add_custom_target(more-inputs)
add_subdirectory(avi)
add_subdirectory(meshes)
add_subdirectory(random)
add_subdirectory(road)
add_subdirectory(scalefree)
