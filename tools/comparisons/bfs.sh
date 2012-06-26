#!/bin/bash
#
# Compare performance with Schardl BFS program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

if [[ ! -e ${BASE}/tools/bin/bfs-schardl ]]; then
  echo "Execute make more-tools before running this script" 1>&2
  exit 1
fi

F=$1

if [[ -z "$F" ]]; then
  echo "usage: $(basename $0) <graph.gr> [args]" 1>&2
  echo "For a useful comparison, graph file and scale must be the same."
  echo "args are passed to Galois programs. A useful example: -t 2" 1>&2
  exit 1
fi

shift
shift

run() {
  cmd="$@"
  echo -en '\033[1;31m'
  echo -n "$cmd"
  echo -e '\033[0m'
  $cmd
}

run ${BASE}/tools/bin/bfs-schardl $(dirname $F)/$(basename $F .gr).bsml 
run ${BASE}/apps/bfs/bfs -parallelInline $* $F
