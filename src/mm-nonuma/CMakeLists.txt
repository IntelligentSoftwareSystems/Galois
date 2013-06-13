add_definitions(-DGALOIS_FORCE_NO_NUMA)
add_internal_library(mm-nonuma ../mm/Mem.cpp ../mm/NumaMem.cpp ../mm/PageAlloc.cpp)

