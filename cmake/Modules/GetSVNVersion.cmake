add_custom_target(svnversion ALL DEPENDS ${PROJECT_BINARY_DIR}/include/Galois/svnversion.h)

find_file(_MODULE "GetSVNVersion-write.cmake" PATHS ${CMAKE_MODULE_PATH})

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/include/Galois/svnversion.h
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -P ${_MODULE})

set_source_files_properties(${PROJECT_BINARY_DIR}/include/Galois/svnversion.h
  PROPERTIES GENERATED TRUE
  HEADER_FILE_ONLY TRUE)
