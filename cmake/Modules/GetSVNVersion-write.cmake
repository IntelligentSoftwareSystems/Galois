### Don't include directly, for use by GetSVNVersion.cmake
find_package(Subversion)
# Extract svn info into MY_XXX variables
if(Subversion_FOUND)
  Subversion_WC_INFO(${SOURCE_DIR} MY)
  if (Subversion_FOUND)
    file(WRITE include/galois/svnversion.h.txt "#define GALOIS_SVNVERSION ${MY_WC_REVISION}\n")
  else()
    file(WRITE include/galois/svnversion.h.txt "#define GALOIS_SVNVERSION 0\n")
  endif()
else()
  file(WRITE include/galois/svnversion.h.txt "#define GALOIS_SVNVERSION 0\n")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different include/galois/svnversion.h.txt include/galois/svnversion.h)
