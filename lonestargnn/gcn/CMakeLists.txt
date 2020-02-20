SET(OPENBLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(OPENBLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)
include_directories(${OPENBLAS_INC})
link_directories(${OPENBLAS_LIB})

app(gcn gcn.cpp)
target_link_libraries(gcn deepgalois)

