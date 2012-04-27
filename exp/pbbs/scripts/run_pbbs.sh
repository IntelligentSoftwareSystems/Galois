#!/bin/bash

THREADS=1
ROUNDS=1

while getopts "t:r:" opt; do
  case $opt in
    t) THREADS=$OPTARG;;
    r) ROUNDS=$OPTARG;;
    \?) exit 1 ;;
    :) exit 1 ;;
  esac
done

shift $((OPTIND-1))

export OMP_NUM_THREADS=$THREADS
export OMP_SCHEDULE="dynamic,16"
export TBB_NUM_THREADS=$THREADS
export CILK_NWORKERS=$THREADS
export GALOIS_NUM_THREADS=$THREADS
export EXP_NUM_ROUNDS=$ROUNDS

PROG=$1

shift

echo "INFO: CommandLine $PROG $*"
echo "INFO: Hostname $(hostname)"
exec $PROG $*
