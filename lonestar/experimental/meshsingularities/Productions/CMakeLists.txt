SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

add_subdirectory(Point2D)
add_subdirectory(Point2DQuad)
add_subdirectory(Point3D)
add_subdirectory(Edge2D)

#SET(MUMPS_MODE "NO" CACHE STRING "Mumps type")
SET(USE_BLAS OFF CACHE BOOL "Use blas")
SET(USE_PAPI_MESH OFF CACHE BOOL "Use PAPI in mesh singularities solver")
#SET(PAPI "NO" CACHE STRING "Use PAPI")

SET(VIRT_ENV "/h1/kjopek/")

SET(BLAS_INC /h1/kjopek/atlas/include/)
SET(BLAS_LIB /h1/kjopek/atlas/lib/)

#SET(MPI_INC ${VIRT_ENV}/include/)
#SET(MPI_LIB ${VIRT_ENV}/lib/)

#SET(METIS_LIB ${VIRT_ENV}/metis/)

#SET(MUMPS ${VIRT_ENV}/MUMPS/)

SET(PAPI_INC ${VIRT_ENV}/papi/include)
SET(PAPI_LIB ${VIRT_ENV}/papi/lib)

include_directories(${BLAS_INC})
include_directories(${VIRT_ENV}/include)
include_directories(${MPI_INC})
include_directories(${MUMPS}/include)

link_directories(${BLAS_LIB})
link_directories(${VIRT_ENV}/lib)
link_directories(${MPI_LIB})
link_directories(${METIS_LIB})
link_directories(${MUMPS}/lib)

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

app(Solver EquationSystem.cpp PointProduction.cpp EdgeProduction.cpp Vertex.cpp Functions.cpp PointSingularitySolver.cpp GaloisWorker.cpp Node.cpp MatrixGeneration/GenericMatrixGenerator.cxx Point3D/TripleArgFunction.cxx Point3D/Element.cxx Point3D/MatrixGenerator.cxx Point3D/Tier.cxx Point2D/DoubleArgFunction.cxx Point2D/Element.cxx Point2D/MatrixGenerator.cxx Point2D/Tier.cxx Edge2D/Tier.cxx Edge2D/MatrixGenerator.cxx FakeMatrixGenerator.cpp)
target_link_libraries(Solver -ldl -lpthread)
if (USE_PAPI_MESH)
    target_link_libraries(Solver -lpapi)
endif()
if (USE_BLAS)
    target_link_libraries(Solver -lcblas -lclapack -latlas -lsatlas -ltatlas)
endif()
