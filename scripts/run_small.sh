#!/bin/bash
#
# Small test cases for each benchmark

# Die on first failed command
set -e 

BASE="$(cd $(dirname $0); cd ..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

SKIPPED=0

run() {
  cmd="$@ -t 2"
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

run apps/avi/AVIodgOrdered -n 0 -d 2 -f "${BASE}/inputs/avi/squareCoarse.NEU.gz"
run apps/barneshut/barneshut -n 1000 -steps 1 -seed 0
run apps/betweennesscentrality/betweennesscentrality "${BASE}/inputs/structured/torus5.gr" -forceVerify
run apps/bfs/bfs "${BASE}/inputs/structured/rome99.gr"
run apps/boruvka/boruvka "${BASE}/inputs/structured/rome99.gr"
run apps/clustering/clustering -numPoints 1000
run apps/delaunayrefinement/delaunayrefinement "${BASE}/inputs/meshes/r10k.1"
run apps/delaunaytriangulation/delaunaytriangulation "${BASE}/inputs/meshes/r10k.node"
run apps/des/DESunordered "${BASE}/inputs/des/multTree6bit.net"
run apps/gmetis/gmetis "${BASE}/inputs/structured/rome99.gr" 4
run apps/kruskal/KruskalHand "${BASE}/inputs/structured/rome99.gr"
run apps/independentset/independentset "${BASE}/inputs/structured/rome99.gr"
run apps/matching/bipartite-mcm 100 1000 10 0
run apps/preflowpush/preflowpush "${BASE}/inputs/structured/srome99.gr" 0 100
run apps/sssp/sssp "${BASE}/inputs/structured/rome99.gr"
run apps/surveypropagation/surveypropagation 9 100 300 3
run apps/tutorial/hello-world 2 10
run apps/tutorial/torus 2 100
run apps/tutorial/torus-improved 2 100

if (($SKIPPED)); then
  echo -en '\033[1;32m'
  echo -n "SOME OK (skipped $SKIPPED)"
  echo -e '\033[0m'
else
  echo -en '\033[1;32m'
  echo -n ALL OK
  echo -e '\033[0m'
fi
