#!/bin/sh
# assumes 2 GPU devices available

MPI=mpiexec
EXEC=$1
INPUT=$2

execname=$(basename "$EXEC" "")
inputdirname=$(dirname "$INPUT")
inputname=$(basename "$INPUT" ".gr")
extension=gr

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=3"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
grep "${inputname}.${extension}" ${source_file}
INPUT=${inputdirname}/${inputname}.${extension}

SET=
if [[ $execname == *"vertex-cut"* ]]; then
  if [[ $inputname == *"road"* ]]; then
    exit
  fi
  # assumes 2 GPU devices available
  SET="gg,2,2 gc,2,2 cg,2,2"
else
  # assumes 2 GPU devices available
  SET="g,1,4 gg,2,2 c,1,4 gc,2,2 cg,2,2"
fi

for task in $SET; do
  IFS=",";
  set $task;
  PFLAGS=$FLAGS
  if [[ $execname == *"vertex-cut"* ]]; then
    PFLAGS+=" -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
  elif [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
    PFLAGS+=" -scalegpu=3"
  fi
  set -x #echo on
  eval "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -comm_mode=2 -noverify |& tee ${execname}_${inputname}_${1}.out"
  set +x #echo off
done

