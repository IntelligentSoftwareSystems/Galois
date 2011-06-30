#!/bin/bash
#
# Compare performance with triangle program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

if [[ ! -e ${BASE}/tools/bin/triangle ]]; then
  echo "Execute make more-tools before running this script" 1>&2
  exit 1
fi

F=$1
if [[ -z "$F" ]]; then
  echo "usage: $(basename $0) <points.node> [args]" 1>&2
  echo "args are passed to Galois programs. A useful example: -t 2" 1>&2
  exit 1
fi

run() {
  cmd="$@"
  echo -en '\033[1;31m'
  echo -n "$cmd"
  echo -e '\033[0m'
  $cmd
}

shift

run ${BASE}/tools/bin/triangle -i -P -N -E -q30 $F
run ${BASE}/apps/delaunaytriangulation/delaunaytriangulation $* $F
M=$(dirname $F)/$(basename $F .node).1
run ${BASE}/apps/delaunayrefinement/delaunayrefinement $* $M
