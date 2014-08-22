set(MUMPS /h1/dgoik/measurements/MUMPS_4.10.0_SEQ/)
include_directories(${MUMPS}include ${MUMPS}libseq)
link_directories(${MUMPS}lib /opt/apps/sysnet/intel/13.1/lib/intel64/ ${MUMPS}libseq/ /usr/lib64/)
add_executable(Mumps_SEQ ${MUMPS}libseq/mpi.h ${MUMPS}include/dmumps_c.h Mumps.cxx  
../MatrixGeneration/GenericMatrixGenerator.cxx ../EquationSystem.cpp
../Point3D/TripleArgFunction.cxx ../Point3D/Element.cxx ../Point3D/MatrixGenerator.cxx ../Point3D/Tier.cxx 
../Point2D/DoubleArgFunction.cxx ../Point2D/Element.cxx ../Point2D/MatrixGenerator.cxx ../Point2D/Tier.cxx)
target_link_libraries(Mumps_SEQ -lifcore -lsvml -ldl libdmumps.a libmumps_common.a libpord.a libmpiseq.a /h1/dgoik/measurements/gilbert_blas/libblas.so libmpiseq.a -lirc -liomp5 -lpthread)

