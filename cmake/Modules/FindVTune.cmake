# Find VTune libraries
# Once done this will define
#  VTune_FOUND - System has VTune
#  VTune_INCLUDE_DIRS - The VTune include directories
#  VTune_LIBRARIES - The libraries needed to use VTune

if(VTune_INCLUDE_DIRS AND VTune_LIBRARIES)
  set(VTune_FIND_QUIETLY TRUE)
endif()

find_path(VTune_INCLUDE_DIRS ittnotify.h PATHS ${VTune_ROOT} PATH_SUFFIXES include)
find_library(VTune_LIBRARY NAMES ittnotify PATHS ${VTune_ROOT} PATH_SUFFIXES lib lib64 lib32)
find_library(VTune_LIBRARIES NAMES dl PATH_SUFFIXES lib lib64 lib32)
set(VTune_LIBRARIES ${VTune_LIBRARY} ${VTune_LIBRARIES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTune DEFAULT_MSG VTune_LIBRARIES VTune_INCLUDE_DIRS)
if(VTUNE_FOUND)
  set(VTune_FOUND on)
endif()
mark_as_advanced(VTune_INCLUDE_DIRS VTune_LIBRARIES)
