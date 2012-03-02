include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(numa numa_allocate "" LIBNUMA_FOUND)
