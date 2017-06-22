#!/bin/sh
# Usage: ./verify.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>
# environment variables: ABELIAN_NON_HETEROGENEOUS ABELIAN_GALOIS_ROOT ABELIAN_EDGE_CUT_ONLY
# executes only on single machine
# assumes 2 GPU devices available (if heterogeneous)

execdirname="."
execname=$1
EXEC=${execdirname}/${execname}

inputdirname=/workspace/dist-inputs
inputname=$2
extension=gr

outputdirname=/workspace/dist-outputs
#outputdirname=/net/ohm/export/cdgc/dist-outputs
IFS='_' read -ra EXECP <<< "$execname"
problem=${EXECP[0]}
OUTPUT=${outputdirname}/${inputname}.${problem}
# kcore output files have a number at the end specifying kcore number
if [[ $execname == *"kcore"* ]]; then
  OUTPUT=${outputdirname}/${inputname}.${problem}100
fi

MPI=mpiexec
LOG=.verify_log

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=3"
fi
# kcore flag
if [[ $execname == *"kcore"* ]]; then
  FLAGS+=" -kcore=100"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
grep "${inputname}.${extension}" ${source_file} >>$LOG
INPUT=${inputdirname}/${inputname}.${extension}

if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/GaloisCpp
fi
checker=${ABELIAN_GALOIS_ROOT}/exp/scripts/result_checker.py

hostname=`hostname`

if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
  # assumes only 2 GPUs device available
  SET="g,1,2 gg,2,2 c,1,16 cc,2,8 cccc,4,4 cccccccc,8,2 gc,2,14 cg,2,14 ggc,3,12 cgg,3,12 gcg,3,12"
else
  SET="c,1,16 cc,2,8 cccc,4,4 cccccccc,8,2"
fi

pass=0
fail=0
failed_cases=""
for partition in 1 2; do
  if [ $partition -eq 2 ]; then
    if [ -z "$ABELIAN_EDGE_CUT_ONLY" ]; then
      FLAGS+=" -enableVertexCut"
    else
      break
    fi
  fi
  for task in $SET; do
    old_ifs=$IFS
    IFS=",";
    set $task;
    PFLAGS=$FLAGS
    if [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
      PFLAGS+=" -scalegpu=3"
    fi
    rm -f output_*.log
    echo "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1 -verify -runs=1" >>$LOG
    eval "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} ${INPUT} -pset=$1 -t=$3 ${PFLAGS} -num_nodes=1 -verify -runs=1" >>$LOG 2>&1
    outputs="output_${hostname}_0.log"
    i=1
    while [ $i -lt $2 ]; do
      outputs+=" output_${hostname}_${i}.log"
      let i=i+1
    done
    eval "sort -n ${outputs} -o output_${hostname}_0.log"
    eval "python $checker $OUTPUT output_${hostname}_0.log &> .output_diff"
    cat .output_diff >> $LOG
    if ! grep -q "SUCCESS" .output_diff ; then
      let fail=fail+1
      if [ $partition -eq 2 ]; then
        failed_cases+="vertex-cut $1 devices with $3 threads; "
      else
        failed_cases+="edge-cut $1 devices with $3 threads; "
      fi
    else
      let pass=pass+1
    fi
    rm .output_diff
    IFS=$old_ifs
  done
done

rm -f output_*.log

echo "---------------------------------------------------------------------------------------"
echo "Algorithm: " $execname
echo "Input: " $inputname
echo $pass "passed test cases"
if [[ $fail == 0 ]] ; then
  echo "Status: SUCCESS"
else
  echo $fail "failed test cases:" $failed_cases
  echo "Status: FAILED"
fi
echo "---------------------------------------------------------------------------------------"

