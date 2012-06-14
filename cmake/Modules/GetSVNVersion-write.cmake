### Don't include directly, for use by GetSVNVersion.cmake
include(FindSubversion)
# Extract svn info into MY_XXX variables
Subversion_WC_INFO(${SOURCE_DIR} MY)
if(Subversion_FOUND)
  file(WRITE include/Galois/svnversion.h.txt "#define SVNVERSION ${MY_WC_REVISION}\n")
else()
  file(WRITE include/Galois/svnversion.h.txt "#define SVNVERSION 0\n")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different include/Galois/svnversion.h.txt include/Galois/svnversion.h)
