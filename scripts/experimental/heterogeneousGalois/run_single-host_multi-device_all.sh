#!/bin/sh
# Usage: ./run_single-host_multi-device_all.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>
# environment variables: ABELIAN_VERIFY ABELIAN_GALOIS_ROOT ABELIAN_VERTEX_CUT ABELIAN_VTUNE
# assumes 2 GPU devices available

execdirname="."
execname=$1
EXEC=${execdirname}/${execname}

inputdirname=/workspace/dist-inputs
inputname=$2
extension=gr

MPI=mpiexec

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -startNode=`cat ${inputdirname}/${inputname}.source`"
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

if [ -n "$ABELIAN_VERIFY" ]; then
  outputdirname=/workspace/dist-outputs
  IFS='_' read -ra EXECP <<< "$execname"
  problem=${EXECP[0]}
  OUTPUT=${outputdirname}/${inputname}.${problem}

  if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
    ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/GaloisCpp
  fi
  checker=${ABELIAN_GALOIS_ROOT}/exp/scripts/result_checker.py

  hostname=`hostname`
fi


if [[ $execname == *"vertex-cut"* ]]; then
  if [[ $inputname == *"road"* ]]; then
    exit
  fi
  # assumes 2 GPU devices available
  SET="gg,2,2 gc,2,14 cg,2,14 ggc,3,12 cgg,3,12 gcg,3,12"
else
  # assumes 2 GPU devices available
  SET="g,1,2 gg,2,2 c,1,16 gc,2,14 cg,2,14 ggc,3,12 cgg,3,12 gcg,3,12"
fi

for task in $SET; do
  IFS=",";
  set $task;
  PFLAGS=$FLAGS
  if [ -n "$ABELIAN_VERTEX_CUT" ]; then
    PFLAGS+=" -enableVertexCut -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
  elif [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
    PFLAGS+=" -scalegpu=3"
  fi
  if [ -n "$ABELIAN_VTUNE" ]; then
    PFLAGS+=" -runs=1"
    CUSTOM_VTUNE="amplxe-cl -collect general-exploration -search-dir /lib/modules/3.10.0-327.22.2.el7.x86_64/weak-updates/nvidia/ -call-stack-mode all -trace-mpi -analyze-system -start-paused -r ${execname}_${inputname}_${1}_exploration"
  fi
  if [ -n "$ABELIAN_VERIFY" ]; then
    PFLAGS+=" -verify"
    rm -f output_*.log
  fi
  echo "GALOIS_DO_NOT_BIND_THREADS=1 $CUSTOM_VTUNE $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1" |& tee ${execname}_${inputname}_${1}.out
  eval "GALOIS_DO_NOT_BIND_THREADS=1 $CUSTOM_VTUNE $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1 |& tee -a ${execname}_${inputname}_${1}.out"
  if [ -n "$ABELIAN_VERIFY" ]; then
    outputs="output_${hostname}_0.log"
    i=1
    while [ $i -lt $2 ]; do
      outputs+=" output_${hostname}_${i}.log"
      let i=i+1
    done
    echo "python $checker $OUTPUT ${outputs}" |& tee -a ${execname}_${inputname}_${1}.out
    eval "python $checker $OUTPUT ${outputs} |& tee -a ${execname}_${inputname}_${1}.out"
  fi
done

rm -f output_*.log

