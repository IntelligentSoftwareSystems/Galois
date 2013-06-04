add_definitions(-DGALOIS_FORCE_NO_NUMA)
file(GLOB Sources
  ${CMAKE_SOURCE_DIR}/src/FileGraph.cpp
  ${CMAKE_SOURCE_DIR}/src/ll/*.cpp
  ${CMAKE_SOURCE_DIR}/src/llvm/*.cpp
  ${CMAKE_SOURCE_DIR}/src/mm/*.cpp
  )
#HACK(ddn): Workaround for bug in Clang-BGQ compiler:
#clang version 3.4
#(hfinkel@login1.mcs.anl.gov:src/llvm-trunk/tools/clang c210ac88e0d229a47343d2162f29f87827f515d1)
#(hfinkel@login1.mcs.anl.gov:src/llvm-trunk 29d31684b696d035f6352c36511267689a0501dd)
if (USE_BGQ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/llvm/CommandLine.cpp PROPERTIES COMPILE_FLAGS "-O0")
endif()
add_library(galois-graph ${Sources})
