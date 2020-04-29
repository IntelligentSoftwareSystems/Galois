# Find OpenBLAS libraries
# Once done this will define
#  OpenBLAS_FOUND - System has OpenBLAS
#  OpenBLAS_INCLUDE_DIRS - The OpenBLAS include directories
#  OpenBLAS_LIBRARIES - The libraries needed to use OpenBLAS

set(OPENBLAS_LIBRARIES) # Include-only library

if(OPENBLAS_INCLUDE_DIRS)
  set(OPENBLAS_FIND_QUIETLY TRUE)
endif()

find_path(OPENBLAS_INCLUDE_DIRS cblas.h PATHS ${OPENBLAS_ROOT} PATH_SUFFIXES include/openblas)
message(STATUS "OPENBLAS_INCLUDE_DIRS: ${OPENBLAS_INCLUDE_DIRS}")
find_library(OPENBLAS_LIBRARY NAMES openblas PATHS ${OPENBLAS_ROOT} PATH_SUFFIXES lib64)
message(STATUS "OPENBLAS_LIBRARY: ${OPENBLAS_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENBLAS DEFAULT_MSG OPENBLAS_LIBRARY OPENBLAS_INCLUDE_DIRS)
if(OPENBLAS_FOUND)
  set(OPENBLAS_FOUND on)
endif()

mark_as_advanced(OPENBLAS_INCLUDE_DIRS)
