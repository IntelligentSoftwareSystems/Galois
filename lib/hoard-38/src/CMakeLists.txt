add_definitions(-O3 -finline-limit=20000 -finline-functions -DNDEBUG -D_REENTRANT=1 -fno-strict-aliasing)
if(NOT USE_PROFILING)
  add_definitions(-fomit-frame-pointer)
endif()

add_definitions(-DDISABLE_OVERLOADING)
include_directories(. heaplayers heaplayers/util)
add_library(hoard gnuwrapper.cpp libhoard.cpp)
target_link_libraries(hoard dl)
