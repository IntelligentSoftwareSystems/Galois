# Find PAPI libraries
# Once done this will define
#  PAPI_FOUND - System has PAPI
#  PAPI_INCLUDE_DIRS - The PAPI include directories
#  PAPI_LIBRARIES - The libraries needed to use PAPI

if(PAPI_INCLUDE_DIRS AND PAPI_LIBRARIES)
  set(PAPI_FIND_QUIETLY TRUE)
endif()

find_path(PAPI_INCLUDE_DIRS papi.h PATHS ${PAPI_ROOT} PATH_SUFFIXES include)
find_library(PAPI_LIBRARY NAMES papi PATHS ${PAPI_ROOT} PATH_SUFFIXES lib lib64)
find_library(PAPI_LIBRARIES NAMES rt PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG PAPI_LIBRARY PAPI_LIBRARIES PAPI_INCLUDE_DIRS)
if(PAPI_FOUND)
  set(PAPI_LIBRARIES ${PAPI_LIBRARY} ${PAPI_LIBRARIES})
endif()

mark_as_advanced(PAPI_INCLUDE_DIRS PAPI_LIBRARIES)
