# Find numa library
# Once done this will define
#  NUMA_FOUND - libnuma found
#  NUMA_OLD - old libnuma API
if(NOT NUMA_FOUND)
  find_library(NUMA_LIBRARIES NAMES numa PATH_SUFFIXES lib lib64)
  if(NUMA_LIBRARIES)
    include(CheckLibraryExists)
    check_library_exists(${NUMA_LIBRARIES} numa_available "" NUMA_FOUND_INTERNAL)
    if(NUMA_FOUND_INTERNAL)
      check_library_exists(${NUMA_LIBRARIES} numa_allocate_nodemask "" NUMA_NEW_INTERNAL)
      if(NOT NUMA_NEW_INTERNAL)
        set(NUMA_OLD "yes" CACHE)
      endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES)
    mark_as_advanced(NUMA_FOUND)
  endif()
endif()

