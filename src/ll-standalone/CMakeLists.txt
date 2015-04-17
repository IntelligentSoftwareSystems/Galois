add_definitions(-DGALOIS_FORCE_STANDALONE)
add_internal_library(ll-standalone ../ll/EnvCheck.cpp ../ll/gIO.cpp ../ll/HWTopo.cpp
  ../ll/HWTopo${hwtopo}.cpp 
  ../ll/SimpleLock.cpp ../ll/PtrLock.cpp ../ll/PtrRWLock.cpp ../ll/TID.cpp)
