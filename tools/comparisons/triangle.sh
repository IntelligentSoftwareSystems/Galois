#!/bin/bash
#
# Compare performance with triangle program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e Makefile ]]; then
  echo "Execute this script from the base of your build directory" 1>&2
  exit 1
fi

if [[ ! -e "${BASE}/tools/bin/triangle" ]]; then
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

run "${BASE}/tools/bin/triangle" -i -P -N -E -q30 "$F"
run "${BASE}/apps/delaunaytriangulation/delaunaytriangulation" $* "$F"
M=$(dirname $F)/$(basename $F .node)
if [[ ! -e "$M.ele" ]]; then
  "${BASE}/apps/delaunaytriangulation/delaunaytriangulation" $F -writemesh "$M"
fi
run "${BASE}/apps/delaunayrefinement/delaunayrefinement" $* "$M"

if [[ ! -e "${BASE}/tools/bin/delaunaytriangulation-pbbs" || ! -e "${BASE}/tools/bin/delaunayrefinement-pbbs" ]]; then
  exit
fi

PBBSF=$(dirname $F)/$(basename $F .node).pbbs.dt
if [[ ! -e "$PBBSF" ]]; then
  echo "Generating PBBS triangulation input from Galois."
  echo "TODO randomize points for best PBBS performance"
  (echo "pbbs_sequencePoint2d"; cat $F | sed '1d' | awk '{print $2,$3;}') > "$PBBSF"
fi
run "${BASE}/tools/bin/delaunaytriangulation-pbbs" "$PBBSF"

PBBSM=$(dirname $F)/$(basename $F .node).pbbs.dmr
if [[ ! -e "$PBBSM" ]]; then
  "${BASE}/tools/bin/delaunaytriangulation-pbbs" -o "$PBBSM" "$PBBSF"
fi
run "${BASE}/tools/bin/delaunayrefinement-pbbs" "$PBBSM"
