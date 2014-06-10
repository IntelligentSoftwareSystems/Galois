add_definitions(-DEXP_DOALL_GALOIS)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorizer-verbose=3")
app(train-ml-galois ../train-ml.cpp ../multilabel.cpp ../wsabie.cpp 
  ../bilinear.cpp ../multiple_linear.cpp ../smat.cpp ../dmat.cpp ../zlib_util.cpp ../tron.cpp
  EXTLIBS ${ZLIB_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
