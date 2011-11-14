file(GLOB Sources *.cpp llvm/*.cpp)

add_library(galois ${Sources})

#target_link_libraries(galois hwloc)
