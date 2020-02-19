SET(USE_BLAS ON CACHE BOOL "Use blas")

SET(BLAS_INC /net/ohm/export/cdgc/cxh/OpenBLAS/build/include)
SET(BLAS_LIB /net/ohm/export/cdgc/cxh/OpenBLAS/build/lib)

if (USE_BLAS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_BLAS")
    include_directories(${BLAS_INC})
    link_directories(${BLAS_LIB})
endif()

app(gcn gcn.cpp)

if (USE_BLAS)
    target_link_libraries(gcn -lopenblas)
endif()
