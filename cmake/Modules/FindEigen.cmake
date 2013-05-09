set(EIGEN_DIR "$ENV{EIGEN_HOME}")

# - Try to find Eigen library
# Once done this will define
#  EIGEN_FOUND - System has Eigen
#  EIGEN_INCLUDE_DIRS - The Eigen include directories
#  EIGEN_LIBRARIES - The libraries needed to use Eigen

set(EIGEN_LIBRARIES) # Include-only library

if(EIGEN_INCLUDE_DIR)
  set(EIGEN_FIND_QUIETLY TRUE)
endif()

find_path(EIGEN_INCLUDE_DIRS NAMES Eigen/Eigen PATHS ${EIGEN_DIR} ${EIGEN_HOME})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set QGLVIEWER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIRS)

mark_as_advanced(EIGEN_INCLUDE_DIRS)
