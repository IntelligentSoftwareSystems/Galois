SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

add_subdirectory(Point2D)
add_subdirectory(Point2DQuad)
add_subdirectory(Point3D)
add_subdirectory(Edge2D)

SET(MUMPS_MODE "NO" CACHE STRING "Mumps type")
SET(BLAS "NO" CACHE STRING "Use blas")
SET(PAPI "NO" CACHE STRING "Use PAPI")

SET(VIRT_ENV "/h1/kjopek/")

SET(BLAS_INC ${VIRT_ENV}/include/)
SET(BLAS_LIB ${VIRT_ENV}/lib/)

SET(MPI_INC ${VIRT_ENV}/include/)
SET(MPI_LIB ${VIRT_ENV}/lib/)

SET(METIS_LIB ${VIRT_ENV}/metis/)

SET(MUMPS ${VIRT_ENV}/MUMPS/)

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

if (${PAPI} STREQUAL "YES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_PAPI")
    include_directories(${PAPI_INC})
    link_directories(${PAPI_LIB})
endif()

if( ${MUMPS_MODE} STREQUAL "SEQ" )
	message("Sequential mumps")
	include_directories(${MUMPS}libseq)
	link_directories(${MUMPS}lib /opt/apps/sysnet/intel/13.1/lib/intel64/ ${MUMPS}libseq)
	
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
	
	if ( ${BLAS} STREQUAL "YES")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_BLAS")
		include_directories(${BLAS_INC})
		link_directories(${BLAS_LIB})
	endif()

	#/usr/lib64/libblas.so.3.2	
	#/h1/dgoik/measurements/gilbert_blas/libdblas.so
	app(Solver EquationSystem.cpp PointProduction.cpp EdgeProduction.cpp
          Vertex.cpp Functions.cpp PointSingularitySolver.cpp GaloisWorker.cpp
          Node.cpp MatrixGeneration/GenericMatrixGenerator.cxx
          Point3D/TripleArgFunction.cxx Point3D/Element.cxx
          Point3D/MatrixGenerator.cxx Point3D/Tier.cxx
          Point2D/DoubleArgFunction.cxx Point2D/Element.cxx
          Point2D/MatrixGenerator.cxx Point2D/Tier.cxx Edge2D/Tier.cxx
          Edge2D/MatrixGenerator.cxx FakeMatrixGenerator.cpp
          EXTLIBS 	dl libpord.a ${BLAS_LIB}/libf77blas.a ${BLAS_LIB}/libcblas.a ${BLAS_LIB}/libatlas.a pthread openmp)
elseif (${MUMPS_MODE} STREQUAL "PAR")
	message("parallel mumps")
	set(MUMPS_PAR /h1/dgoik/measurements/MUMPS_4.10.0_PAR/)
	set(LIBS_DIR /h1/dgoik/measurements/libs/)
	set(LAM_2 /h1/kjopek/lam2/)
	include_directories(${LAM_2}include/ ${MUMPS_PAR}include)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

	
	if ( ${BLAS} STREQUAL "YES")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_BLAS")
		include_directories(${BLAS_INC})
		link_directories(${BLAS_LIB})
	endif()
	
	link_directories(${MUMPS_PAR}lib ${LIBS_DIR} /opt/apps/sysnet/intel/13.1/lib/intel64/ /usr/lib64/ /usr/local/lib ${LAM_2}lib)
	app(Solver ${LAM_2}include/mpi.h ${MUMPS_PAR}include/dmumps_c.h
          Mumps_Par.cxx EquationSystem.cpp PointProduction.cpp
          EdgeProduction.cpp Vertex.cpp Functions.cpp
          PointSingularitySolver.cpp GaloisWorker.cpp Node.cpp
          MatrixGeneration/GenericMatrixGenerator.cxx
          Point3D/TripleArgFunction.cxx Point3D/Element.cxx
          Point3D/MatrixGenerator.cxx Point3D/Tier.cxx
          Point2D/DoubleArgFunction.cxx Point2D/Element.cxx
          Point2D/MatrixGenerator.cxx Point2D/Tier.cxx Edge2D/Tier.cxx
          Edge2D/MatrixGenerator.cxx FakeMatrixGenerator.cpp
          EXTLIBS ifcore svml libdmumps.a libmumps_common.a libpord.a scalapack /usr/lib64/atlas/libf77blas.so.3 /usr/lib64/atlas/libcblas.so.3 mpi lammpi++ lammpio lamf77mpi lam util d irc iomp5 pthread ${BLAS_LIB}/libcblas.a ${BLAS_LIB}/liblapack.a)	
else()
	app(Solver EquationSystem.cpp PointProduction.cpp EdgeProduction.cpp
          Vertex.cpp Functions.cpp PointSingularitySolver.cpp GaloisWorker.cpp
          Node.cpp MatrixGeneration/GenericMatrixGenerator.cxx
          Point3D/TripleArgFunction.cxx Point3D/Element.cxx
          Point3D/MatrixGenerator.cxx Point3D/Tier.cxx
          Point2D/DoubleArgFunction.cxx Point2D/Element.cxx
          Point2D/MatrixGenerator.cxx Point2D/Tier.cxx Edge2D/Tier.cxx
          Edge2D/MatrixGenerator.cxx FakeMatrixGenerator.cpp DynamicLib.cpp
          EXTLIBS dl pthread papi)
endif()
