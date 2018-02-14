#!/bin/sh

if [ -z "$ABELIAN_LLVM_BUILD" ]; then
  ABELIAN_LLVM_BUILD=/net/velocity/workspace/SourceCode/llvm/build
fi
if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/Galois
fi
#if [ -z "$ABELIAN_GALOIS_BUILD" ]; then
#  ABELIAN_GALOIS_BUILD=/net/velocity/workspace/SourceCode/Galois/build/verify
#fi
  if [ -z "$ABELIAN_GGC_ROOT" ]; then
    ABELIAN_GGC_ROOT=/net/velocity/workspace/SourceCode/ggc
  fi
MPI_INCLUDE=/opt/apps/ossw/libraries/mpich2/mpich2-3.1.4/c7/gcc-4.9/include

echo "Using LLVM build:" $ABELIAN_LLVM_BUILD
echo "Using Galois:" $ABELIAN_GALOIS_ROOT
  echo "Using GGC:" $ABELIAN_GGC_ROOT

CXX_DEFINES="-DBOOST_NO_AUTO_PTR -DGALOIS_COPYRIGHT_YEAR=2015 -DGALOIS_VERSION=2.3.0 -DGALOIS_VERSION_MAJOR=2 -DGALOIS_VERSION_MINOR=3 -DGALOIS_VERSION_PATCH=0 -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS"
CXX_FLAGS="-g -Wall -gcc-toolchain $GCC_BIN/.. -fopenmp -fcolor-diagnostics -O3 -DNDEBUG -I$ABELIAN_GALOIS_ROOT/libdist/include -I$ABELIAN_GALOIS_ROOT/dist_apps/include -I$MPI_INCLUDE -I$BOOST_INC -I$ABELIAN_GALOIS_ROOT/lonestar/include -I$ABELIAN_GALOIS_ROOT/libgalois/include -I$ABELIAN_GALOIS_ROOT/libruntime/include -I$ABELIAN_GALOIS_ROOT/libnet/include -I$ABELIAN_GALOIS_ROOT/libllvm/include -std=gnu++14"

GGC_FLAGS="--cuda-worklist basic --cuda-graph basic --opt parcomb --opt np --npf 8 "
#GGC_FLAGS="--cuda-worklist basic --cuda-graph basic --opt parcomb"
#GGC_FLAGS+=" --loglevel DEBUG"
if [ -f "GGCFLAGS" ]; then
  GGC_FLAGS+=$(head -n 1 "GGCFLAGS")
fi
echo "Using GGC FLAGS:" $GGC_FLAGS

CXX=$ABELIAN_LLVM_BUILD/bin/clang++
  IRGL_CXX="$CXX -Xclang -load -Xclang $ABELIAN_LLVM_BUILD/lib/GaloisFunctions.so -Xclang -plugin -Xclang irgl"
  GGC="$ABELIAN_GGC_ROOT/src/ggc"

log=.log

echo "Cleaning generated files"
if [ -n "$1" ]; then
  rm -f $log gen_cuda.cu
else
  rm -f $log gen_cuda.py gen_cuda.cu gen_cuda.cuh gen_cuda.h
fi

if ! [ -n "$1" ]; then
  echo "Generating IrGL code"
  $IRGL_CXX $CXX_DEFINES $CXX_FLAGS -o .temp.o -c gen.cpp >>$log 2>&1
fi
  echo "Generating CUDA code from IrGL"
  $GGC $GGC_FLAGS -o gen_cuda.cu gen_cuda.py >>$log 2>&1

if [ -n "$1" ]; then
  echo "Generated files: gen_cuda.cu" 
else
  echo "Generated files: gen_cuda.py gen_cuda.h gen_cuda.cuh gen_cuda.cu" 
fi

rm -f Entry-*.dot cdep_Entry-*.dot

