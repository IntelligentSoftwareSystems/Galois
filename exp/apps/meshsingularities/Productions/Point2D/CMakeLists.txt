#app(PrintMatrix2D main.cpp MatrixGenerator.cxx DoubleArgFunction.cxx Element.cxx Tier.cxx)
#app(Chol2D Cholesky.cpp MatrixGenerator.cxx DoubleArgFunction.cxx Element.cxx)

#set(MUMPS /h1/dgoik/measurements/MUMPS_4.10.0_SEQ/)
#include_directories(${MUMPS}include ${MUMPS}libseq)
#link_directories(${MUMPS}lib /opt/apps/sysnet/intel/13.1/lib/intel64/ ${MUMPS}libseq/ /usr/lib64/)
#add_executable(Mumps2D_SEQ ${MUMPS}libseq/mpi.h ${MUMPS}include/dmumps_c.h Mumps.cxx MatrixGenerator.cxx DoubleArgFunction.cxx Element.cxx Tier.cxx ../EquationSystem.cpp)
#target_link_libraries(Mumps2D_SEQ -lifcore -lsvml -ldl libdmumps.a libmumps_common.a libpord.a libmpiseq.a /h1/dgoik/measurements/gilbert_blas/libblas.so libmpiseq.a -lirc -liomp5 -lpthread)


app(Point2DTest main.cpp MatrixGenerator.cxx DoubleArgFunction.cxx Element.cxx Tier.cxx ../EquationSystem.cpp)