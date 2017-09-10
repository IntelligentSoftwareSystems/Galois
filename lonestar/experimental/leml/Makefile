CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -g -fPIC -fopenmp -std=gnu++0x
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp  -std=gnu++0x
BLASFLAGS = -llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib

# ======================== flags for ICC on stampede ===========================
#CXX = icc
#CC = icc
#ICC_CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=gnu++0x
#ICC_BLAS = -I$MKLROOT/include -mkl=parallel -openmp 
#BLASFLAGS = $(ICC_BLAS)
#CFLAGS = $(ICC_CFLAGS)

# ======================== flags for G++ on stampede ===========================
# module swap intel gcc; module load mkl 
#CXX = g++ 
#CC = gcc
#GCC_CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=gnu++0x
#GCC_BLAS = -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
#BLASFLAGS = ${GCC_BLAS}
#CFLAGS = ${GCC_CFLAGS}



LIBSFLAGS= $(BLASFLAGS) -lz -Lzlib 

LIBS = blas/blas.a
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

ALLLIB = multilabel.o wsabie.o bilinear.o multiple_linear.o smat.o dmat.o tron.o blas/blas.a zlib_util.o zlib/libz.a

#all: train-ml  test-smat test-blas
all: train-ml 
lib: $(ALLLIB)

tar: 
	tar cvzf v4-full.tgz *.cpp *.h Makefile blas zlib matlab

test-smat: test-smat.cpp smat.o
	$(CXX) $(CFLAGS) -o test-smat test-smat.cpp $(ALLLIB) $(LIBSFLAGS) 

test-blas: test-blas.cpp
	$(CXX) $(CFLAGS) -o test-blas test-blas.cpp $(LIBSFLAGS)

train-ml: train-ml.cpp multilabel.o wsabie.o bilinear.o multiple_linear.o smat.o dmat.o zlib_util.o tron.o $(ALLLIB)
	$(CXX) $(CFLAGS) -o train-ml train-ml.cpp $(ALLLIB) $(LIBSFLAGS) 

multilabel.o: multilabel.cpp multilabel.h wsabie.h bilinear.h multiple_linear.h tron.h smat.h dmat.h wsabie.h
	$(CXX) $(CFLAGS) -c -o multilabel.o multilabel.cpp  $(LIBSFLAGS)

wsabie.o: wsabie.cpp wsabie.h smat.h dmat.h
	$(CXX) $(CFLAGS) -c -o wsabie.o wsabie.cpp  $(LIBSFLAGS)

bilinear.o: bilinear.cpp bilinear.h tron.h smat.h
	$(CXX) $(CFLAGS) -c -o bilinear.o bilinear.cpp  $(LIBSFLAGS)

multiple_linear.o: multiple_linear.h multiple_linear.cpp
	${CXX} ${CFLAGS} -c -o multiple_linear.o multiple_linear.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

smat.o: smat.h smat.cpp 
	${CXX} ${CFLAGS} -c -o smat.o smat.cpp

dmat.o: dmat.h dmat.cpp 
	${CXX} ${CFLAGS} -c -o dmat.o dmat.cpp

zlib_util.o: zlib_util.h zlib_util.cpp
	${CXX} ${CFLAGS} -c -o zlib_util.o zlib_util.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

zlib/libz.a: 
	make -C zlib

train: tron.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

clean:
	rm -f *~ tron.o linear.o train predict bilinear.o util.o liblinear.so.$(SHVER) smat.o multiple_linear.o multilabel.o
	rm -f *~ *.o train-ml 
	rm -f test-smat test-blas
	make -C blas clean
	make -C zlib clean
	make -C matlab clean
