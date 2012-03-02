include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(numa numa_available "" LIBNUMA_FOUND)
