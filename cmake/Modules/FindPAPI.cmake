set(PAPI_DIR "/h1/lenharth/papi")

# - Try to find PAPI
# Once done this will define
#  PAPI_FOUND - System has PAPI
#  PAPI_INCLUDE_DIRS - The PAPI include directories
#  PAPI_LIBRARIES - The libraries needed to use PAPI

find_path(PAPI_INCLUDE_DIR papi.h PATHS ${PAPI_DIR} PATH_SUFFIXES include)
find_library(PAPI_LIBRARY NAMES papi PATHS ${PAPI_DIR} PATH_SUFFIXES lib lib64)

set(PAPI_LIBRARIES ${PAPI_LIBRARY} )
set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PAPI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PAPI  DEFAULT_MSG
                                  PAPI_LIBRARY PAPI_INCLUDE_DIR)

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY )