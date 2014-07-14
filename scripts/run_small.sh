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

if [ python -c "import sys;sys.exit(0 if sys.byteorder=='big' else 1)" ]; then
  ROME="${BASE}/inputs/structured/rome99.gr"
  SROME="${BASE}/inputs/structured/srome99.gr"
else
  ROME="${BASE}/inputs/structured/rome99.big.gr"
  SROME="${BASE}/inputs/structured/srome99.big.gr"
fi

runSmallTest() {
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

runSmallTest apps/avi/AVIodgExplicitNoLock -n 0 -d 2 -f "${BASE}/inputs/avi/squareCoarse.NEU.gz"
runSmallTest apps/barneshut/barneshut -n 1000 -steps 1 -seed 0
runSmallTest apps/betweennesscentrality/betweennesscentrality-outer "${BASE}/inputs/structured/torus5.gr" -forceVerify
runSmallTest apps/bfs/bfs "${ROME}"
runSmallTest apps/boruvka/boruvka "${ROME}"
runSmallTest apps/clustering/clustering -numPoints 1000
runSmallTest apps/delaunayrefinement/delaunayrefinement "${BASE}/inputs/meshes/r10k.1"
runSmallTest apps/delaunaytriangulation/delaunaytriangulation "${BASE}/inputs/meshes/r10k.node"
runSmallTest apps/des/DESunordered "${BASE}/inputs/des/multTree6bit.net"
runSmallTest apps/gmetis/gmetis "${ROME}" 4
runSmallTest apps/kruskal/KruskalHand "${ROME}"
runSmallTest apps/independentset/independentset "${ROME}"
runSmallTest apps/matching/bipartite-mcm -inputType generated -n 100 -numEdges 1000 -numGroups 10 -seed 0
runSmallTest apps/preflowpush/preflowpush "${SROME}" 0 100
runSmallTest apps/sssp/sssp "${ROME}"
runSmallTest apps/surveypropagation/surveypropagation 9 100 300 3
runSmallTest apps/tutorial/hello-world 2 10
runSmallTest apps/tutorial/torus 2 100
runSmallTest apps/tutorial/torus-improved 2 100

if (($SKIPPED)); then
  echo -en '\033[1;32m'
  echo -n "SOME OK (skipped $SKIPPED)"
  echo -e '\033[0m'
else
  echo -en '\033[1;32m'
  echo -n ALL OK
  echo -e '\033[0m'
fi
