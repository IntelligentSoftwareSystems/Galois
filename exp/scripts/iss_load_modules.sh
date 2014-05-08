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
  module load cmake
else 
  # Load system gcc first so that cmake can be found
  module load gcc/4.8
  module load cmake
  module load gcc/4.9.0-scale
fi

module load tbb
module load boost
module load eigen
module load neon
module load subversion

if [ "$1" != "min" ]; then
  module load clang/3.4-noconflict
  module load gdb
  module load mkl
  module load mpich2
  module load git
fi
