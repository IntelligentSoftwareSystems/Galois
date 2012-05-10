file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp)

add_library(galois ${Sources})
if(USE_EXP)
  if(EXP_DOALL MATCHES "DMP")
    target_link_libraries(galois dmpruntime)
  endif()
endif()
