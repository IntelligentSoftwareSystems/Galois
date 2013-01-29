#!/bin/bash
#
# Compare performance with PBBS minimal spanning tree program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

if [[ ! -e "${BASE}/tools/bin/kruskal-pbbs" ]]; then
  echo "Execute make more-tools before running this script" 1>&2
  exit 1
fi

F=$1

if [[ -z "$F" ]]; then
  echo "usage: $(basename $0) <graph.gr> [args]" 1>&2
  echo "args are passed to Galois programs. A useful example: -t 2" 1>&2
  exit 1
fi

shift

run() {
  cmd="$@"
  echo -en '\033[1;31m'
  echo -n "$cmd"
  echo -e '\033[0m'
  $cmd
}

PBBSF=$(dirname $F)/$(basename $F .gr).pbbs
if [[ ! -e "$PBBSF" ]]; then
  "${BASE}/tools/graph-convert/graph-convert" -gr2intpbbs "$F" "$PBBSF"
fi
run "${BASE}/tools/bin/kruskal-pbbs" "$PBBSF"

run "${BASE}/apps/kruskal/KruskalHand" $* "$F"
