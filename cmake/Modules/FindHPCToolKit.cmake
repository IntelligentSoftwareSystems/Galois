# Find HPCToolKit libraries
# Once done this will define
#  HPCToolKit_FOUND - System has lib
#  HPCToolKit_INCLUDE_DIRS - The include directories
#  HPCToolKit_LIBRARIES - The libraries needed to use

if(HPCToolKit_INCLUDE_DIRS AND HPCToolKit_LIBRARIES)
  set(HPCToolKit_FIND_QUIETLY TRUE)
endif()

find_path(HPCToolKit_INCLUDE_DIRS hpctoolkit.h PATHS ${HPCToolKit_ROOT} PATH_SUFFIXES include)
find_library(HPCToolKit_LIBRARY NAMES libhpctoolkit.a hpctoolkit PATHS ${HPCToolKit_ROOT} PATH_SUFFIXES lib lib64 lib32 lib/hpctoolkit)
set(HPCToolKit_LIBRARIES ${HPCToolKit_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HPCToolKit DEFAULT_MSG HPCToolKit_LIBRARIES HPCToolKit_INCLUDE_DIRS)
if(HPCTOOLKIT_FOUND)
  set(HPCToolKit_FOUND on)
endif()
mark_as_advanced(HPCToolKit_INCLUDE_DIRS HPCToolKit_LIBRARIES)
