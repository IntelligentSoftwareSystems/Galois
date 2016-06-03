#!/bin/sh

MPI=mpiexec
EXEC=$1
INPUT=$2

execname=$(basename "$EXEC" "")
inputdirname=$(dirname "$INPUT")
inputname=$(basename "$INPUT" ".gr")
if [[ $EXEC == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  inputname=${inputname}.transpose
  INPUT=${inputdirname}/${inputname}.gr
fi

FLAGS=

SET=
if [[ $EXEC == *"vertex-cut"* ]]; then
  if [[ $INPUT == *"road"* ]]; then
    exit
  fi
  SET="gc,2,2 cg,2,2 gg,2,2 gggc,4,1"
else
  SET="c,1,4 g,1,4 gc,2,2 cg,2,2 gg,2,2 ggg,3,1 gggc,4,1"
fi

for task in $SET; do
  IFS=",";
  set $task;
  PFLAGS=$FLAGS
  if [[ $EXEC == *"vertex-cut"* ]]; then
    PFLAGS+=" -partFolder=${inputdirname}/partitions/${2}/${inputname}.gr"
  else
    if [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
      PFLAGS+=" -scalegpu=3"
    fi
  fi
  set -x #echo on
  eval "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} -noverify ${PFLAGS} -pset=$1 -t=$3 ${INPUT} |& tee ${execname}_${inputname}_${1}.out"
  set +x #echo off
done

