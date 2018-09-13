#!/bin/sh
# Usage: ./run_single-host_multi-device_all.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>
# environment variables: ABELIAN_VERIFY ABELIAN_GALOIS_ROOT ABELIAN_VERTEX_CUT ABELIAN_VTUNE
# assumes 4 GPU devices available

execdir="."
execname=$1
EXEC=${execdir}/${execname}

#inputdirname=/workspace/dist-inputs
inputdirname=/net/ohm/export/iss/dist-inputs
inputname=$2
extension=gr

MPI=mpiexec

FLAGS=
# kcore flag
if [[ $execname == *"kcore"* ]]; then
  # TODO: update this for non-100 kcore numbers
  FLAGS+=" -kcore=100"
fi
if [[ ($execname == *"bc"*) || ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -startNode=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ ($execname == *"bc"*) ]]; then
  FLAGS+=" -singleSource"
fi
#if [[ ($execname == *"pagerank"*) ]]; then
#  FLAGS+=" -maxIterations=100"
#fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* || $execname == *"kcore"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
  FLAGS+=" -symmetricGraph"
else 
  # for verify purposes, always pass in graph transpose just in case it is 
  # needed for non-symmetric graphs
  FLAGS+=" -graphTranspose=${inputdirname}/transpose/${inputname}.tgr"
fi
INPUT=${inputdirname}/${inputname}.${extension}

if [ -n "$ABELIAN_VERIFY" ]; then
  #outputdirname=/workspace/dist-outputs
  outputdirname=/net/ohm/export/iss/dist-outputs
  IFS='_' read -ra EXECP <<< "$execname"
  problem=${EXECP[0]}
  OUTPUT=${outputdirname}/${inputname}.${problem}

  if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
    ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/Galois
  fi
  checker=${ABELIAN_GALOIS_ROOT}/scripts/result_checker.py

  hostname=`hostname`
fi

# assumes 2 GPU devices available
SET="g,1,2 gg,2,2 c,1,16 gc,2,14 cg,2,14 ggc,3,12 cgg,3,12 gcg,3,12"
# assumes 6 GPU devices available - tuxedo
SET="c,1,48 g,1,2 gg,2,2 ggg,3,2 gggg,4,2 ggggg,5,2 gggggg,6,2"
# assumes 4 GPU devices available
SET="c,1,28 g,1,28 gg,2,14 ggg,3,7 gggg,4,7"
SET="gggg,4,7"

for task in $SET; do
  IFS=",";
  set $task;
  PFLAGS=$FLAGS
  statname=${execname}_${inputname}_${1}.stats
  PFLAGS+=" -statFile=${execdir}/${statname}"
  if [ -n "$ABELIAN_VERTEX_CUT" ]; then
    PFLAGS+=" -partition=cvc"
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
  rm -f ${execname}_${inputname}_${1}.out
  grep "${inputname}.${extension}" ${source_file} |& tee ${execname}_${inputname}_${1}.out
  echo "GALOIS_DO_NOT_BIND_THREADS=1 $CUSTOM_VTUNE $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1" |& tee ${execname}_${inputname}_${1}.out
  eval "GALOIS_DO_NOT_BIND_THREADS=1 $CUSTOM_VTUNE $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1 |& tee -a ${execname}_${inputname}_${1}.out"
  if [ -n "$ABELIAN_VERIFY" ]; then
    outputs="output_${hostname}_0.log"
    i=1
    while [ $i -lt $2 ]; do
      outputs+=" output_${hostname}_${i}.log"
      let i=i+1
    done
    echo "python $checker -t=1 $OUTPUT ${outputs}" |& tee -a ${execname}_${inputname}_${1}.out
    eval "python $checker -t=1 $OUTPUT ${outputs} |& tee -a ${execname}_${inputname}_${1}.out"
  fi
done

rm -f output_*.log

