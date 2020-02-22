SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})
if(USE_CPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
endif()

app(gcn gcn.cpp)
target_link_libraries(gcn deepgalois)

