set(MUMPS_PAR /h1/dgoik/measurements/MUMPS_4.10.0_PAR/)
set(LIBS_DIR /h1/dgoik/measurements/libs/)
set(LAM_2 /h1/dgoik/measurements/lam2/)
include_directories(${LAM_2}include/ ${MUMPS_PAR}include)
link_directories(${MUMPS_PAR}lib ${LIBS_DIR} /opt/apps/sysnet/intel/13.1/lib/intel64/ /usr/lib64/ /usr/local/lib ${LAM_2}lib)
add_executable(Mumps_PAR ${LAM_2}/include/mpi.h ${MUMPS_PAR}include/dmumps_c.h Mumps_par.cxx
../MatrixGeneration/GenericMatrixGenerator.cxx ../EquationSystem.cpp
../Point3D/TripleArgFunction.cxx ../Point3D/Element.cxx ../Point3D/MatrixGenerator.cxx ../Point3D/Tier.cxx 
../Point2D/DoubleArgFunction.cxx ../Point2D/Element.cxx ../Point2D/MatrixGenerator.cxx ../Point2D/Tier.cxx)
target_link_libraries(Mumps_PAR -lifcore -lsvml libdmumps.a libmumps_common.a libpord.a -lscalapack /h1/dgoik/measurements/libs/blacs1.a /h1/dgoik/measurements/libs/blacs2.a
-llammpio -llamf77mpi -lmpi -llam -lutil -ldl /h1/dgoik/measurements/gilbert_blas/libblas.so -lirc -liomp5 -lpthread)
