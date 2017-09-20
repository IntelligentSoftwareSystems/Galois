# DUMMY is a non-existent file to force regeneration of svn header every build
add_custom_target(revision ALL DEPENDS DUMMY ${PROJECT_BINARY_DIR}/include/galois/revision.h)

find_file(_MODULE "GetGitVersion-write.cmake" PATHS ${CMAKE_MODULE_PATH})

add_custom_command(OUTPUT DUMMY ${PROJECT_BINARY_DIR}/include/galois/revision.h
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
  -DCMAKE_MODULE_PATH="${CMAKE_SOURCE_DIR}/cmake/Modules/" -P ${_MODULE})

set(_MODULE off)

set_source_files_properties(${PROJECT_BINARY_DIR}/include/galois/revision.h
  PROPERTIES GENERATED TRUE
  HEADER_FILE_ONLY TRUE)
