# Makefile for simple (non-Galois) Cholesky.
#
# Some versions of the algorithm use BLAS and/or LAPACK. You will need
# to update your BLAS flags below. FIXME: Switch to Galois CMake or
# something.

CC=gcc
CXX=g++

DEBUG=-g -Wall -DNDEBUG
OPTIMIZE=-O3
ifdef BOOST_INC
CPPFLAGS=-I$(BOOST_INC)
endif
CFLAGS=$(DEBUG) $(OPTIMIZE) $(CPPFLAGS)
CXXFLAGS=$(DEBUG) $(OPTIMIZE) $(CPPFLAGS) -std=c++11
GENFLAGS=-MMD
LIBS=-lm -lpthread

# PAPI configuration
PAPI_INC=-I/usr/lib64/papi-5.1.1/usr/include -DPAPI
PAPI_LIB=-L/usr/lib64/papi-5.1.1/usr/lib -lpapi

# BLAS configuration
BLAS_INC=-I/net/nordstrom/workspace/ATLAS/include
BLAS_LIBINC=-L/net/nordstrom/workspace/ATLAS/Linux_`hostname -s`/lib
BLAS_CLIB=$(BLAS_LIBINC) -llapack -lcblas -latlas
BLAS_FLIB=$(BLAS_LIBINC) -lf77blas -latlas

all: dagcholesky dagcholesky.serial leftcholesky leftcholesky.serial upcholesky.serial supernodalcholesky.serial

clean:
	rm -f *.d *.o dagcholesky dagcholesky.serial leftcholesky leftcholesky.serial upcholesky upcholesky.serial upcholesky-orig.serial supernodalcholesky supernodalcholesky.serial

-include *.d

%.o: %.cpp Makefile
	$(CXX) $(GENFLAGS) -c -o $@ $< $(CXXFLAGS)
%.o: %.c Makefile
	$(CC) $(GENFLAGS) -c -o $@ $< $(CFLAGS)

dagcholesky.serial: LIBS += $(PAPI_LIB) $(BLAS_CLIB)
dagcholesky.serial: CPPFLAGS+=$(PAPI_INC) $(BLAS_INC)
dagcholesky.serial: dagcholesky.cpp
	$(CXX) -DSERIAL $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)
dagcholesky: LIBS += $(BLAS_CLIB)
dagcholesky: CPPFLAGS += $(BLAS_INC)
dagcholesky: dagcholesky.cpp
	$(CXX)          $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)


leftcholesky.serial: LIBS += $(PAPI_LIB)
leftcholesky.serial: CPPFLAGS+=$(PAPI_INC)
leftcholesky.serial: leftcholesky.cpp
	$(CXX) -DSERIAL $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)
leftcholesky: leftcholesky.cpp
	$(CXX)          $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)

upcholesky.serial: LIBS += $(PAPI_LIB)
upcholesky.serial: CPPFLAGS+=$(PAPI_INC)
upcholesky.serial: upcholesky.cpp
	$(CXX) -DSERIAL $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)
#upcholesky: LIBS += -fopenmp
#upcholesky: upcholesky.cpp
#	$(CXX)          $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)

supernodalcholesky.serial: LIBS += $(PAPI_LIB)
supernodalcholesky.serial: CPPFLAGS+=$(PAPI_INC)
supernodalcholesky.serial: LIBS += $(BLAS_CLIB)
supernodalcholesky.serial: CPPFLAGS += $(BLAS_INC)
supernodalcholesky.serial: supernodalcholesky.cpp
	$(CXX) -DSERIAL $(GENFLAGS) -o $@ $< $(CXXFLAGS) $(LIBS)
