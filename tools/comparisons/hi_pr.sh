#!/bin/bash
#
# Compare performance with triangle program

BASE="$(cd $(dirname $0); cd ../..; pwd)"

if [[ ! -e ${BASE}/tools/bin/hi_pr ]]; then
  echo "Execute make more-tools before running this script" 1>&2
  exit 1
fi

G=$1
SOURCE=$2
SINK=$3
if [[ -z "$SINK" ]]; then
  echo "usage: $(basename $0) <graph.gr> <source node> <sink node> [args]" 1>&2
  echo "args are passed to Galois programs. A useful example: -t 2" 1>&2
  exit 1
fi

D="$(dirname $G)/$(basename $G .gr).dimacs"
if [[ ! -e "$G" || ! -e "$D" ]]; then
  echo "Missing gr file or dimacs file." 1>&2
  echo "Use graph-convert to make one from the other." 1>&2
  exit 1
fi

run() {
  cmd="$@"
  echo -en '\033[1;31m'
  echo -n "$cmd"
  echo -e '\033[0m'
  $@
}

shift 3

run echo hi_pr
HSOURCE=$(($SOURCE + 1))
HSINK=$(($SINK + 1))
# Bash tokenizing rules are complicated :(
perl -p -e "s/p sp (\d+) (\d+)/p max \1 \2\nn $HSOURCE s\nn $HSINK t/;" $D \
  | ${BASE}/tools/bin/hi_pr

if [[ -e ${BASE}/apps/preflowpush/preflowpush ]]; then
  run ${BASE}/apps/preflowpush/preflowpush $* $G $SOURCE $SINK
fi
