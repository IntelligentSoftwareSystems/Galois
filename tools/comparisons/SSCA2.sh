#!/bin/bash
#
# Compare performance with SSCA2 program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

if [[ ! -e ${BASE}/tools/bin/SSCA2 ]]; then
  echo "Execute make more-tools before running this script" 1>&2
  exit 1
fi

F=$1
S=$2
if [[ -z "$S" ]]; then
  echo "usage: $(basename $0) <graph.gr> <scale> [args]" 1>&2
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

run ${BASE}/tools/bin/SSCA2 $S
run ${BASE}/apps/betweennesscentrality/betweennesscentrality $* $F
