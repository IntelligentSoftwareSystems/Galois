set(QGLVIEWER_DIR)

# - Try to find QGLViewer
# Once done this will define
#  QGLVIEWER_FOUND - System has QGLViewer
#  QGLVIEWER_INCLUDE_DIRS - The QGLViewer include directories
#  QGLVIEWER_LIBRARIES - The libraries needed to use QGLViewer

if(QGLVIEWER_INCLUDE_DIR AND QGLVIEWER_LIBRARIES)
  set(QGLVIEWER_FIND_QUIETLY TRUE)
endif()

find_path(QGLVIEWER_INCLUDE_DIRS NAMES QGLViewer/qglviewer.h)
find_library(QGLVIEWER_LIBRARIES NAMES QGLViewer PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set QGLVIEWER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(QGLVIEWER DEFAULT_MSG QGLVIEWER_INCLUDE_DIRS QGLVIEWER_LIBRARIES)

mark_as_advanced(QGLVIEWER_INCLUDE_DIRS QGLVIEWER_LIBRARIES)
