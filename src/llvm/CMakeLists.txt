add_internal_library(llvm Allocator.cpp APFloat.cpp APInt.cpp
  CommandLine.cpp SmallPtrSet.cpp SmallVector.cpp StringMap.cpp StringRef.cpp
  Twine.cpp)

#HACK(ddn): Workaround for bug in Clang-BGQ compiler:
#clang version 3.4
#(hfinkel@login1.mcs.anl.gov:src/llvm-trunk/tools/clang c210ac88e0d229a47343d2162f29f87827f515d1)
#(hfinkel@login1.mcs.anl.gov:src/llvm-trunk 29d31684b696d035f6352c36511267689a0501dd)
if(CMAKE_SYSTEM_NAME MATCHES "BlueGeneQ" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(CommandLine.cpp PROPERTIES COMPILE_FLAGS "-O0")
endif()
