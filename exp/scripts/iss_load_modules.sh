#!/bin/bash

if [ "$(basename $0)" == "iss_load_modules.sh" ]; then
  echo "Source this file instead of running directly" >&2
  exit 1
fi

module load sl6
module load cmake
module load boost/1.52
module load lapack

if [ "$1" != "min" ]; then
  module load gdb/7.5
  module load gcc/4.7.1-scale
  #module load tbb
  module load vtune
  module load mkl
fi
