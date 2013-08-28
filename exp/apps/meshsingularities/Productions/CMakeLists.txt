app(Test2D Test2D.cpp EquationSystem.cpp Production.cpp Vertex.cpp Processing.cpp Functions.cpp Point2D/DoubleArgFunction.cxx Point2D/Element.cxx Point2D/MatrixGenerator.cxx Point2D/Tier.cxx MatrixGeneration/GenericMatrixGenerator.cxx)

add_subdirectory(Point2D)
add_subdirectory(Point2DQuad)
add_subdirectory(Point3D)

SET(MUMPS "SEQ" CACHE STRING "Mumps type")
if( ${MUMPS} STREQUAL "SEQ" )
	message("Sequential mumps")
	set(MUMPS /h1/dgoik/measurements/MUMPS_4.10.0_SEQ/)
	include_directories(${MUMPS}include ${MUMPS}libseq)
	link_directories(${MUMPS}lib /opt/apps/sysnet/intel/13.1/lib/intel64/ ${MUMPS}libseq/ /usr/lib64/)
	
	app(Solver EquationSystem.cpp Production.cpp Vertex.cpp GraphGenerator.cxx Processing.cpp Functions.cpp PointSingularitySolver.cpp GaloisWorker.cpp MatrixGeneration/GenericMatrixGenerator.cxx Point3D/TripleArgFunction.cxx Point3D/Element.cxx Point3D/MatrixGenerator.cxx Point3D/Tier.cxx Point2D/DoubleArgFunction.cxx Point2D/Element.cxx Point2D/MatrixGenerator.cxx Point2D/Tier.cxx Mumps.cxx)
	target_link_libraries(Solver -lifcore -lsvml -ldl libdmumps.a libmumps_common.a libpord.a libmpiseq.a /h1/dgoik/measurements/gilbert_blas/libblas.so libmpiseq.a -lirc -liomp5 -lpthread)
elseif (${MUMPS} STREQUAL "PAR")
	message("parallel mumps")
	set(MUMPS_PAR /h1/dgoik/measurements/MUMPS_4.10.0_PAR/)
	set(LIBS_DIR /h1/dgoik/measurements/libs/)
	set(LAM_2 /h1/dgoik/measurements/lam2/)
	include_directories(${LAM_2}include/ ${MUMPS_PAR}include)
	link_directories(${MUMPS_PAR}lib ${LIBS_DIR} /opt/apps/sysnet/intel/13.1/lib/intel64/ /usr/lib64/ /usr/local/lib ${LAM_2}lib)
	app(Solver ${LAM_2}include/mpi.h ${MUMPS_PAR}include/dmumps_c.h Mumps_Par.cxx EquationSystem.cpp Production.cpp Vertex.cpp GraphGenerator.cxx Processing.cpp Functions.cpp PointSingularitySolver.cpp GaloisWorker.cpp MatrixGeneration/GenericMatrixGenerator.cxx Point3D/TripleArgFunction.cxx Point3D/Element.cxx Point3D/MatrixGenerator.cxx Point3D/Tier.cxx Point2D/DoubleArgFunction.cxx Point2D/Element.cxx Point2D/MatrixGenerator.cxx Point2D/Tier.cxx)
	target_link_libraries(Solver -lifcore -lsvml libdmumps.a libmumps_common.a libpord.a -lscalapack /h1/dgoik/measurements/libs/blacs1.a /h1/dgoik/measurements/libs/blacs2.a
	-llammpio -llamf77mpi -lmpi -llam -lutil -ldl /h1/dgoik/measurements/gilbert_blas/libblas.so -lirc -liomp5 -lpthread)	
endif()

