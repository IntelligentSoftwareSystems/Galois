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

RESULTDIR="$(pwd)/weblogs"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

SKIPPED=0

run() {
  cmd="$@ -noverify"
  name=$(echo "$1" | sed -e 's/\//_/g')
  logfile="$RESULTDIR/$name.log"
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

run apps/avi/AVIodgExplicitNoLock -d 2 -n 1 -e 0.1 -f "$BASEINPUT/avi/10x10_42k.NEU.gz"
run apps/barneshut/barneshut -n 50000 -steps 1 -seed 0
run apps/betweennesscentrality/betweennesscentrality-outer "$BASEINPUT/scalefree/rmat8-2e14.gr"
run apps/bfs/bfs "$BASEINPUT/random/r4-2e26.gr"
run apps/boruvka/boruvka "$BASEINPUT/road/USA-road-d.USA.gr"
run apps/clustering/clustering -numPoints 10000
run apps/delaunayrefinement/delaunayrefinement "$BASEINPUT/meshes/r5M"
run apps/delaunaytriangulation/delaunaytriangulation "$BASEINPUT/meshes/r5M.node"
run apps/des/DESunordered "$BASEINPUT/des/koggeStone64bit.net"
run apps/gmetis/gmetis "$BASEINPUT/road/USA-road-d.USA.gr" 256
run apps/kruskal/KruskalHand -maxRounds 600 -lowThresh 16 -preAlloc 32 "$BASEINPUT/random/r4-2e24.gr"
run apps/independentset/independentset "$BASEINPUT/random/r4-2e26.gr"
run apps/matching/bipartite-mcm -inputType generated -n 1000000 -numEdges 100000000 -numGroups 10000 -seed 0
run apps/preflowpush/preflowpush "$BASEINPUT/random/r4-2e23.gr" 0 100
run apps/spanningtree/spanningtree "$BASEINPUT/random/r4-2e26.gr"
run apps/sssp/sssp -delta 8 "$BASEINPUT/random/r4-2e26.gr"
run apps/surveypropagation/surveypropagation 9 1000000 3000000 3

# Generate results
#cat $RESULTDIR/*.log | python "$BASE/scripts/report.py" > report.csv
#Rscript "$BASE/scripts/report.R" report.csv report.json

if (($SKIPPED)); then
  echo -en '\033[1;32m'
  echo -n "SOME OK (skipped $SKIPPED)"
  echo -e '\033[0m'
else
  echo -en '\033[1;32m'
  echo -n ALL OK
  echo -e '\033[0m'
fi
