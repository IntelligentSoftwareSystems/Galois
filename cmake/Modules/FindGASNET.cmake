# Find the GasNet librairy
#  GASNET_FOUND - system has GasNet lib
#  GASNET_INCLUDE_DIR - the GasNet include directory
#  GASNET_LIBRARIES - Libraries needed to use GasNet

if(GASNET_INCLUDE_DIRS AND GASNET_LIBRARIES)
  set(GASNET_FIND_QUIETLY TRUE)
endif()

find_path(GASNET_INCLUDE_DIRS NAMES gasnet.h)
find_library(GASNET_LIBRARY_1 NAMES gasnet amudp HINTS ${GASNET_INCLUDE_DIRS}/../lib )
find_library(GASNET_LIBRARY_2 NAMES gasnet gasnet-udp-par HINTS ${GASNET_INCLUDE_DIRS}/../lib )

set(GASNET_LIBRARIES ${GASNET_LIBRARY_2} ${GASNET_LIBRARY_1})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GASNET DEFAULT_MSG GASNET_INCLUDE_DIRS GASNET_LIBRARIES)

mark_as_advanced(GASNET_INCLUDE_DIRS GASNET_LIBRARIES)
