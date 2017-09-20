### Don't include directly, for use by GetSVNVersion.cmake
find_package(Git)
# Extract svn info into MY_XXX variables
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --verify --short HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  file(WRITE include/galois/revision.h.txt "#define GALOIS_REVISION \"${GIT_REVISION}\"\n")
else()
  file(WRITE include/galois/revision.h.txt "#define GALOIS_REVISION \"0\"\n")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different include/galois/revision.h.txt include/galois/revision.h)
