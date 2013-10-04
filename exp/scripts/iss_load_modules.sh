#!/bin/bash

if [ "$(basename -- $0)" == "iss_load_modules.sh" ]; then
  echo "Source this file instead of running directly" >&2
  exit 1
fi

# first up remove everything
module purge

module load sl6
module use /net/faraday/workspace/local/modules/modulefiles
module load lapack
module load vtune

if [ "$1" == "intel" ]; then
  module load intel
else 
  module load gcc/4.8.1-scale
fi

module load cmake
module load tbb
module load boost
module load eigen
module load neon
module load subversion

if [ "$1" != "min" ]; then
  module load clang/3.3-noconflict
  module load gdb
  module load mkl
  module load mpich2
fi




