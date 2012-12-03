if(USE_EXP)
  file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp ../exp/src/*.cpp)
else()
  file(GLOB Sources *.cpp llvm/*.cpp mm/*.cpp ll/*.cpp)
endif()

add_library(galois ${Sources})

target_link_libraries(galois ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(galois ${ZLIB_LIBRARIES})
target_link_libraries(galois ${Boost_LIBRARIES})

if(LIBNUMA_FOUND)
	target_link_libraries(galois numa)
endif()

if(VTUNE_FOUND)
	target_link_libraries(galois ${VTUNE_LIBRARIES})
endif()

if(PAPI_FOUND)
	target_link_libraries(galois ${PAPI_LIBRARIES})
endif()

if(TBB_FOUND)
	target_link_libraries(galois ${TBB_LIBRARIES})
endif()


