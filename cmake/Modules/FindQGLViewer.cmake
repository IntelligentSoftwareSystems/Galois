# Find QGLViewer libraries
# Once done this will define
#  QGLViewer_FOUND - System has QGLViewer
#  QGLViewer_INCLUDE_DIRS - The QGLViewer include directories
#  QGLViewer_LIBRARIES - The libraries needed to use QGLViewer

if(QGLViewer_INCLUDE_DIRS AND QGLVIEWER_LIBRARIES)
  set(QGLViewer_FIND_QUIETLY TRUE)
endif()

find_path(QGLViewer_INCLUDE_DIRS NAMES QGLViewer/qglviewer.h)
find_library(QGLViewer_LIBRARIES NAMES QGLViewer PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QGLViewer DEFAULT_MSG QGLViewer_INCLUDE_DIRS QGLViewer_LIBRARIES)
if(QGLVIEWER_FOUND)
  set(QGLViewer_FOUND TRUE)
endif()

mark_as_advanced(QGLViewer_INCLUDE_DIRS QGLViewer_LIBRARIES)
