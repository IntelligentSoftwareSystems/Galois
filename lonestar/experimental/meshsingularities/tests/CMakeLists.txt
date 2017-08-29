SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

SET(VIRT_ENV "/h1/kjopek/")

SET(BLAS_INC /h1/kjopek/atlas/include/)
SET(BLAS_LIB /h1/kjopek/atlas/lib/)

include_directories(${BLAS_INC})
link_directories(${BLAS_LIB})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_BLAS")

app(BlasTest BlasTest.cpp)
target_link_libraries(BlasTest -lcblas -llapack -latlas -lsatlas -ltatlas)
