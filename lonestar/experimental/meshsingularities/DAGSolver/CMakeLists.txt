SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")


SET(USE_BLAS OFF CACHE BOOL "Use blas")
SET(USE_PAPI_MESH OFF CACHE BOOL "Use PAPI in mesh singularities solver")

SET(VIRT_ENV "/h1/kjopek/")

SET(BLAS_INC /h1/kjopek/atlas/include/)
SET(BLAS_LIB /h1/kjopek/atlas/lib/)

SET(PAPI_INC ${VIRT_ENV}/papi/include)
SET(PAPI_LIB ${VIRT_ENV}/papi/lib)

if (USE_PAPI_MESH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_PAPI")
    include_directories(${PAPI_INC})
    link_directories(${PAPI_LIB})
endif()

if (USE_BLAS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_BLAS")
    include_directories(${BLAS_INC})
    link_directories(${BLAS_LIB})
endif()

app(DAGSolver DAGSolver.cpp DynamicLib.cpp EquationSystem.cpp Analysis.cpp Node.cpp Mesh.cpp CILK.cpp Seq.cpp GaloisDag.cpp)
target_link_libraries(DAGSolver -ldl -lpthread)
if (USE_PAPI_MESH)
    target_link_libraries(DAGSolver -lpapi)
endif()
if (USE_BLAS)
    target_link_libraries(DAGSolver -lcblas -llapack -latlas -lsatlas -ltatlas)
endif()
