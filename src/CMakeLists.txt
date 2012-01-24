file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp)

add_library(galois ${Sources})

#target_link_libraries(galois hwloc)
