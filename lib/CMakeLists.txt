#add_subdirectory(libhugetlbfs-2.10)
if(USE_SOLARIS)
  message("Hoard not supported on solaris (yet)")
else()
  add_subdirectory(hoard-38)
endif()

#include(ExternalProject)
#
#ExternalProject_Add(
#   hwloc-1-1-1
#   URL ${CMAKE_CURRENT_SOURCE_DIR}/hwloc-1.1.1.tar.gz
#   INSTALL_DIR ${CMAKE_BINARY_DIR}/lib/install
#   CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-static=true
#   BUILD_COMMAND make
#)
#
#set_property(TARGET hwloc PROPERTY DEPENDS hwloc-1-1-1)

