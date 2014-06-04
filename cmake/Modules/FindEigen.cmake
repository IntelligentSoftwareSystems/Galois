# Find Eigen library
# Once done this will define
#  Eigen_FOUND - System has Eigen
#  Eigen_INCLUDE_DIRS - The Eigen include directories
#  Eigen_LIBRARIES - The libraries needed to use Eigen

set(Eigen_LIBRARIES) # Include-only library

if(Eigen_INCLUDE_DIR)
  set(Eigen_FIND_QUIETLY TRUE)
endif()

find_path(Eigen_INCLUDE_DIRS NAMES Eigen/Eigen PATHS ENV EIGEN_HOME PATHS ENV EIGEN_INCLUDE)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Eigen_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Eigen DEFAULT_MSG Eigen_INCLUDE_DIRS)
if(EIGEN_FOUND)
  set(Eigen_FOUND TRUE)
endif()

mark_as_advanced(Eigen_INCLUDE_DIRS)
