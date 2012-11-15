if(USE_EXP)
  file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp ../exp/src/*.cpp)
else()
  file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp)
endif()

add_library(galois ${Sources})
