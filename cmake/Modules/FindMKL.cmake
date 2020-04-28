# Find MKL libraries
# Once done this will define
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - The MKL include directories
#  MKL_LIBRARIES - The libraries needed to use MKL

set(MKL_LIBRARIES) # Include-only library

if(MKL_INCLUDE_DIRS)
  set(MKL_FIND_QUIETLY TRUE)
endif()

find_path(MKL_INCLUDE_DIRS mkl.h PATHS ${MKL_ROOT} PATH_SUFFIXES include)
message(STATUS "MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
find_library(MKL_LIBRARY NAMES mkl_rt PATHS ${MKL_ROOT} PATH_SUFFIXES lib/intel64)
message(STATUS "MKL_LIBRARY: ${MKL_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARY MKL_INCLUDE_DIRS)
if(MKL_FOUND)
  set(MKL_FOUND on)
endif()

mark_as_advanced(MKL_INCLUDE_DIRS)
