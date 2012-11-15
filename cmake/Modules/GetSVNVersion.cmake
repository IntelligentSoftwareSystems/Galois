# DUMMY is a non-existent file to force regeneration of svn header every build
add_custom_target(svnversion ALL DEPENDS DUMMY ${PROJECT_BINARY_DIR}/include/Galois/svnversion.h)

find_file(_MODULE "GetSVNVersion-write.cmake" PATHS ${CMAKE_MODULE_PATH})

add_custom_command(OUTPUT DUMMY ${PROJECT_BINARY_DIR}/include/Galois/svnversion.h
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -P ${_MODULE})

set(_MODULE off)

set_source_files_properties(${PROJECT_BINARY_DIR}/include/Galois/svnversion.h
  PROPERTIES GENERATED TRUE
  HEADER_FILE_ONLY TRUE)
