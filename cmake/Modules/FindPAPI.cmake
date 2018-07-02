# Find PAPI libraries
# Once done this will define
#  PAPI_FOUND - System has PAPI
#  PAPI_INCLUDE_DIRS - The PAPI include directories
#  PAPI_LIBRARIES - The libraries needed to use PAPI

if(PAPI_INCLUDE_DIRS AND PAPI_LIBRARIES)
  set(PAPI_FIND_QUIETLY TRUE)
endif()

# XXX(ddn): our system papi is broken so ignore for now
# find_path(PAPI_INCLUDE_DIRS papi.h HINTS ${PAPI_ROOT} PATH_SUFFIXES include NO_DEFAULT_PATH )
find_path(PAPI_INCLUDE_DIRS papi.h HINTS ${PAPI_ROOT} ENV TACC_PAPI_DIR PATH_SUFFIXES include)
message(STATUS "PAPI_INCLUDE_DIRS: ${PAPI_INCLUDE_DIRS}")
find_library(PAPI_LIBRARY NAMES papi HINTS ${PAPI_ROOT} ENV TACC_PAPI_DIR PATH_SUFFIXES lib lib64)
message(STATUS "PAPI_LIBRARY: ${PAPI_LIBRARY}")
find_library(PAPI_LIBRARIES NAMES rt PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG PAPI_LIBRARY PAPI_LIBRARIES PAPI_INCLUDE_DIRS)
if(PAPI_FOUND)
  set(PAPI_LIBRARIES ${PAPI_LIBRARY} ${PAPI_LIBRARIES})
endif()

mark_as_advanced(PAPI_INCLUDE_DIRS PAPI_LIBRARIES)
