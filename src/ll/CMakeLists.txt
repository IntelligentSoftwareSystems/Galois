set(hwtopo)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(hwtopo "Linux")
elseif(CMAKE_SYSTEM_NAME MATCHES "BlueGeneQ")
  set(hwtopo "BlueGeneQ")
elseif(CMAKE_SYSTEM_NAME MATCHES "(Solaris|SunOS)")
  set(hwtopo "Solaris")
else()
  message(FATAL_ERROR "Unknown system name: ${CMAKE_SYSTEM_NAME}")
endif()

add_internal_library(ll EnvCheck.cpp gIO.cpp HWTopo.cpp
  HWTopo${hwtopo}.cpp SimpleLock.cpp
  TID.cpp)
