add_subdirectory(apps)
add_subdirectory(src)
add_subdirectory(test)

set(USE_PBBS OFF CACHE BOOL "Compile pbbs subtree")
if(USE_PBBS)
  add_subdirectory(pbbs)
endif()
