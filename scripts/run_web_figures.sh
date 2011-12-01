#!/bin/bash
#
# Larger test cases for each benchmark

set -e

BASE="$(cd $(dirname $0); cd ..; pwd)"
if [[ -z "$BASEINPUT" ]]; then
  BASEINPUT="$(pwd)/inputs"
fi

if [[ -z "$NUMTHREADS" ]]; then
  NUMTHREADS="$(cat /proc/cpuinfo | grep processor | wc -l)"
fi

RESULTDIR="$(pwd)/logs"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

SKIPPED=0

run() {
  cmd="$@"
  shortname=$(echo "$cmd" | sed -e 's/apps\/\([^/]*\)\/.*/\1/')
  logfile="$RESULTDIR/$shortname.log"
  if [[ ! -e "$logfile"  && -x "$1" ]]; then
    echo -en '\033[1;31m'
    echo -n "$cmd"
    echo -e '\033[0m'
    $BASE/scripts/run.py --threads 1:$NUMTHREADS --timeout $((60*20)) -- $cmd 2>&1 | tee "$logfile"
  else
    echo -en '\033[1;31m'
    echo -n Skipping $1
    echo -e '\033[0m'
    SKIPPED=$(($SKIPPED + 1))
  fi
}

mkdir -p "$RESULTDIR"

run apps/avi/AVIunordered -noverify -d 2 -n 1 -e 0.1 -f "$BASEINPUT/avi/10x10_42k.NEU"
run apps/clustering/clustering -numPoints 10000
run apps/barneshut/barneshut -noverify -n 50000 -steps 1 -seed 0
run apps/betweennesscentrality/betweennesscentrality "$BASEINPUT/scalefree/rmat13.gr"
run apps/boruvka/boruvka "$BASEINPUT/road/USA-road-d.USA.gr"
run apps/delaunayrefinement/delaunayrefinement "$BASEINPUT/meshes/r5M"
run apps/delaunaytriangulation/delaunaytriangulation "$BASEINPUT/meshes/r5M.node"
run apps/des/DESunordered -noverify -epi 512 "$BASEINPUT/des/koggeStone64bit.net"
run apps/gmetis/gmetis -mtxinput "$BASEINPUT/matrix/cage15.mtx" 256
run apps/preflowpush/preflowpush "$BASEINPUT/random/r4-2e23.gr" 0 100
run apps/sssp/sssp -delta 14 "$BASEINPUT/random/r4-2e25.gr"
run apps/surveypropagation/surveypropagation 9 1000000 3000000 3

if (($SKIPPED)); then
  echo -en '\033[1;32m'
  echo -n "SOME OK (skipped $SKIPPED)"
  echo -e '\033[0m'
else
  echo -en '\033[1;32m'
  echo -n ALL OK
  echo -e '\033[0m'
fi
