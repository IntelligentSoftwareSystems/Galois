app(NumericCholesky NumericCholesky.cpp)
app(UpCholesky UpCholesky.cpp)
app(UpCholeskySimpleGraph UpCholeskySimpleGraph.cpp)

if (CBLAS_FOUND)
  app(SuperNodalCholesky SuperNodalCholesky.cpp)
endif()
