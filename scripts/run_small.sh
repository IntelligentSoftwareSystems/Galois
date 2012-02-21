#!/bin/bash
#
# Small test cases for each benchmark

set -e

BASE="$(cd $(dirname $0); cd ..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

SKIPPED=0

run() {
  cmd="$@"
  if [[ -x $1 ]]; then
    echo -en '\033[1;31m'
    echo -n "$cmd"
    echo -e '\033[0m'
    $cmd
  else
    echo -en '\033[1;31m'
    echo -n Skipping $1
    echo -e '\033[0m'
    SKIPPED=$(($SKIPPED + 1))
  fi
}

run apps/avi/AVIunordered -t 2 -n 0 -d 2 -f ${BASE}/inputs/avi/squareCoarse.NEU
run apps/barneshut/barneshut -t 2 -n 1000 -steps 1 -seed 0
run apps/betweennesscentrality/betweennesscentrality -t 2 ${BASE}/inputs/structured/torus5.gr
run apps/boruvka/boruvka -t 2 ${BASE}/inputs/structured/rome99.gr
run apps/clustering/clustering -t 2 -numPoints 1000
run apps/delaunaytriangulation/delaunaytriangulation -t 2 ${BASE}/inputs/meshes/r10k.node
run apps/delaunayrefinement/delaunayrefinement -t 2 ${BASE}/inputs/meshes/r10k.1
run apps/des/DESunordered -t 2 ${BASE}/inputs/des/multTree6bit.net 
run apps/gmetis/gmetis -t 2 ${BASE}/inputs/structured/rome99.gr 4 #false true false
run apps/surveypropagation/surveypropagation -t 2 9 100 300 3
run apps/preflowpush/preflowpush -t 2 ${BASE}/inputs/structured/rome99.gr 0 100
run apps/sssp/sssp -t 2 ${BASE}/inputs/structured/rome99.gr

if (($SKIPPED)); then
  echo -en '\033[1;32m'
  echo -n "SOME OK (skipped $SKIPPED)"
  echo -e '\033[0m'
else
  echo -en '\033[1;32m'
  echo -n ALL OK
  echo -e '\033[0m'
fi
