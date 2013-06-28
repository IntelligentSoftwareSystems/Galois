# Find PAPI libraries
# Once done this will define
#  PAPI_FOUND - System has PAPI
#  PAPI_INCLUDE_DIRS - The PAPI include directories
#  PAPI_LIBRARIES - The libraries needed to use PAPI

set(PAPI_DIR "/h1/lenharth/papi")

if(PAPI_INCLUDE_DIRS AND PAPI_INCLUDE_DIRS)
  set(PAPI_FIND_QUIETLY TRUE)
endif()

find_path(PAPI_INCLUDE_DIRS papi.h PATHS ${PAPI_DIR} PATH_SUFFIXES include)
find_library(PAPI_LIBRARIES NAMES papi PATHS ${PAPI_DIR} PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG PAPI_LIBRARIES PAPI_INCLUDE_DIRS)

mark_as_advanced(PAPI_INCLUDE_DIRS PAPI_LIBRARIES)
