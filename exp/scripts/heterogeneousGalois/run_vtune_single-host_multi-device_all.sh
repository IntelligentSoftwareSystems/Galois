#!/bin/sh
# assumes 3 GPU devices available

MPI=mpiexec
EXEC=$1
INPUT=$2

execname=$(basename "$EXEC" "")
inputdirname=$(dirname "$INPUT")
inputname=$(basename "$INPUT" ".gr")
extension=gr
if [[ $EXEC == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
if [[ $EXEC == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
fi
INPUT=${inputdirname}/${inputname}.${extension}

FLAGS=

SET=
if [[ $EXEC == *"vertex-cut"* ]]; then
  if [[ $INPUT == *"road"* ]]; then
    exit
  fi
  # assumes 3 GPU devices available
  SET="gc,2,2 cg,2,2 gg,2,2 gggc,4,1"
else
  # assumes 3 GPU devices available
  SET="c,1,4 g,1,4 gc,2,2 cg,2,2 gg,2,2"
fi

for task in $SET; do
  IFS=",";
  set $task;
  PFLAGS=$FLAGS
  if [[ $EXEC == *"vertex-cut"* ]]; then
    PFLAGS+=" -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
  else
    if [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
      PFLAGS+=" -scalegpu=3"
    fi
  fi
  set -x #echo on
  eval "GALOIS_DO_NOT_BIND_THREADS=1 amplxe-cl -collect general-exploration -search-dir /lib/modules/3.10.0-327.22.2.el7.x86_64/weak-updates/nvidia/ -call-stack-mode all -trace-mpi -analyze-system -start-paused -r ${execname}_${inputname}_${1}_exploration $MPI -n=$3 ${EXEC} ${INPUT} -noverify -pset=$1 ${PFLAGS} -runs=1 -t=$3 |& tee ${execname}_${inputname}_${1}.out"
  eval "GALOIS_DO_NOT_BIND_THREADS=1 amplxe-cl -collect advanced_hotspots -search-dir /lib/modules/3.10.0-327.22.2.el7.x86_64/weak-updates/nvidia/ -call-stack-mode all -trace-mpi -analyze-system -start-paused -r ${execname}_${inputname}_${1}_hotspots $MPI -n=$3 ${EXEC} ${INPUT} -noverify -pset=$1 ${PFLAGS} -runs=1 -t=$3 |& tee ${execname}_${inputname}_${1}.out"
  set +x #echo off
done

