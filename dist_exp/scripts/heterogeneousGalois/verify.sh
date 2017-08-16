#!/bin/sh
# Usage: ./verify.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>
# environment variables: ABELIAN_NON_HETEROGENEOUS ABELIAN_GALOIS_ROOT ABELIAN_EDGE_CUT_ONLY
# executes only on single machine
# assumes 2 GPU devices available (if heterogeneous)

execdirname="."
execname=$1
EXEC=${execdirname}/${execname}

#inputdirname=/net/ohm/export/cdgc/dist-inputs
inputdirname=dist-inputs
inputname=$2
extension=gr

outputdirname=dist-outputs
#outputdirname=/workspace/dist-outputs

IFS='_' read -ra EXECP <<< "$execname"
problem=${EXECP[0]}
OUTPUT=${outputdirname}/${inputname}.${problem}

# kcore output files have a number at the end specifying kcore number
if [[ $execname == *"kcore"* ]]; then
  OUTPUT=${outputdirname}/${inputname}.${problem}100
fi

# for bc, do single source outputs
if [[ ($execname == *"bc"*) ]]; then
  OUTPUT=${outputdirname}/${inputname}.ssbc
fi

# for bc, if using rmat15, then use all sources output (without ss)
# TODO currently even rmat15 uses single source, hence rmat16 which doesn't 
# exist
if [[ ($execname == *"bc"*) && ($inputname == "rmat16") ]]; then
  OUTPUT=${outputdirname}/rmat15.bc
fi

MPI=mpiexec
LOG=.verify_log

#FLAGS=" -maxIterations=200"
#FLAGS+=" -numComputeSubsteps=8"
#FLAGS+=" -numPipelinedPhases=8"
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
  FLAGS+=" -symmetricGraph"
  extension=sgr
else 
  # for verify purposes, always pass in graph transpose just in case it is 
  # needed for non-symmetric graphs
  FLAGS+=" -graphTranspose=${inputdirname}/transpose/${inputname}.tgr"
fi

# bc: if rmat15 is not used, specify single source flags else do
# all sources for rmat15
# TODO currently uses rmat16 (doesn't exist) so everything does single source
if [[ ($execname == *"bc"*) && ! ($inputname == "rmat16") ]]; then
  FLAGS+=" -singleSource"
  FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
fi

grep "${inputname}.${extension}" ${source_file} >>$LOG
INPUT=${inputdirname}/${inputname}.${extension}

if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/GaloisCpp
fi
checker=${ABELIAN_GALOIS_ROOT}/exp/scripts/result_checker.py
#checker=./result_checker.py

hostname=`hostname`

if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
  # assumes only 2 GPUs device available
  #SET="g,1,48 gg,2,24 gggg,4,12 gggggg,6,8 c,1,48 cc,2,24 cccc,4,12 cccccccc,8,6 cccccccccccccccc,16,3"
  SET="g,1,16 gg,2,8 gc,2,8 cg,2,8, ggc,3,4 cgg,3,4 c,1,16 cc,2,8 ccc,3,4 cccc,4,4 ccccc,5,2 cccccc,6,2 ccccccc,7,2 cccccccc,8,2 ccccccccc,9,1 cccccccccc,10,1 ccccccccccc,11,1 cccccccccccc,12,1 ccccccccccccc,13,1 cccccccccccccc,14,1 cccccccccccccc,15,1 ccccccccccccccc,16,1"
else
  #SET="c,1,48 cc,2,24 cccc,4,12 cccccccc,8,6 cccccccccccccccc,16,3"
  SET="c,1,16 cc,2,8 ccc,3,4 cccc,4,4 ccccc,5,2 cccccc,6,2 ccccccc,7,2 cccccccc,8,2 ccccccccc,9,1 cccccccccc,10,1 ccccccccccc,11,1 cccccccccccc,12,1 ccccccccccccc,13,1 cccccccccccccc,14,1 cccccccccccccc,15,1 ccccccccccccccc,16,1"
fi

FLAGS+=" -doAllKind=DOALL_COUPLED_RANGE"
#FLAGS+=" -edgeNuma"

pass=0
fail=0
failed_cases=""
for partition in 1 2 3 4; do
  CUTTYPE=

  if [ $partition -eq 2 ]; then
    CUTTYPE+=" -partition=pl_vcut"
  elif [ $partition -eq 3 ]; then
    CUTTYPE+=" -partition=cart_vcut"
  elif [ $partition -eq 4 ]; then
    CUTTYPE+=" -partition=iec"
  fi

  for task in $SET; do
    old_ifs=$IFS
    IFS=",";
    set $task;
    if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
      PFLAGS="-pset=$1 -num_nodes=1"
    else
      PFLAGS=""
    fi
    PFLAGS+=$FLAGS
    if [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
      PFLAGS+=" -scalegpu=3"
    fi
    rm -f output_*.log

    echo "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} ${INPUT} -t=$3 ${PFLAGS} ${CUTTYPE} -verify" >>$LOG
    eval "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} ${INPUT} -t=$3 ${PFLAGS} ${CUTTYPE} -verify" >>$LOG 2>&1
    #outputs="output_${hostname}_0.log"
    #i=1
    #while [ $i -lt $2 ]; do
    #  outputs+=" output_${hostname}_${i}.log"
    #  let i=i+1
    #done
    #eval "sort -nu ${outputs} -o output_${hostname}_*.log"
    eval "sort -nu output_${hostname}_*.log -o output_${hostname}_0.log"

    # slightly higher threshold for bc + pagerank to reduce prints to log
    if [[ ($execname == *"bc"*) || ($execname == *"pagerank"*) ]]; then
      eval "python $checker -t=0.01 $OUTPUT output_${hostname}_0.log &> .output_diff"
    else
      eval "python $checker $OUTPUT output_${hostname}_0.log &> .output_diff"
    fi


    cat .output_diff >> $LOG
    if ! grep -q "SUCCESS" .output_diff ; then
      let fail=fail+1
      if [ $partition -eq 4 ]; then
        failed_cases+="incoming edge-cut $1 devices with $3 threads; "
      elif [ $partition -eq 3 ]; then
        failed_cases+="cartesian vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 2 ]; then
        failed_cases+="powerlyra vertex-cut $1 devices with $3 threads; "
      else
        failed_cases+="outgoing edge-cut $1 devices with $3 threads; "
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

