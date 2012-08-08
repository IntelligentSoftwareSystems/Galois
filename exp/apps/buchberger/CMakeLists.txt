include_directories(${GMP_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorizer-verbose=2")
app(buchberger REQUIRES GMP_FOUND EXTLIBS ${GMP_LIBRARIES} ${GMPXX_LIBRARIES})
