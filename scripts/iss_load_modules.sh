#!/bin/bash

if [ "$(basename -- $0)" == "iss_load_modules.sh" ]; then
  echo "Source this file instead of running directly" >&2
  exit 1
fi

# first up remove everything
module purge

module use /opt/apps/ossw/modulefiles/

module load c7
module load serf

module use /net/faraday/workspace/local/modules/modulefiles
module use /org/centers/cdgc/modules

module load atc
module load cmake
module load mpich2
module load boost
module load gdb
module load isspython # needed for vim
module load git

if [ "$1" != "min" ]; then
  module load tbb
  module load eigen
  module load neon
  module load lapack
  module load vtune
  module load mkl
  module load texlive
  module load subversion
  # module load screen #disabling for now because screen was compiled without proper color support
  if [ "$SYSTEMTYPE" != "c7" ] ; then
    module load doxygen
    module load gnuplot
    module load ghostscript
    module load valgrind
  fi
fi
