#!/bin/sh
# Usage: ./verify.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>
# environment variables: ABELIAN_NON_HETEROGENEOUS ABELIAN_GALOIS_ROOT ABELIAN_EDGE_CUT_ONLY
# executes only on single machine
# assumes 2 GPU devices available (if heterogeneous)

execdirname="."
execname=$1
EXEC=${execdirname}/${execname}

#inputdirname=/net/ohm/export/cdgc/dist-inputs
inputdirname=/workspace/dist-inputs
inputname=$2
extension=gr

#outputdirname=/net/ohm/export/cdgc/dist-outputs
outputdirname=/workspace/dist-outputs

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

FLAGS=
#FLAGS+=" -doAllKind=DOALL_COUPLED_RANGE"
# kcore flag
if [[ $execname == *"kcore"* ]]; then
  FLAGS+=" -kcore=100"
fi
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
fi

# bc: if rmat15 is not used, specify single source flags else do
# all sources for rmat15
# TODO currently uses rmat16 (doesn't exist) so everything does single source
if [[ ($execname == *"bc"*) && ! ($inputname == "rmat16") ]]; then
  FLAGS+=" -singleSource"
  FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
  FLAGS+=" -symmetricGraph"
else 
  # for verify purposes, always pass in graph transpose just in case it is 
  # needed for non-symmetric graphs
  FLAGS+=" -graphTranspose=${inputdirname}/transpose/${inputname}.tgr"
fi

grep "${inputname}.${extension}" ${source_file} >>$LOG
INPUT=${inputdirname}/${inputname}.${extension}

if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/Galois
fi
checker=${ABELIAN_GALOIS_ROOT}/dist_exp/scripts/result_checker.py
#checker=./result_checker.py

hostname=`hostname`

if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
  # assumes only 2 GPUs device available
  #SET="g,1,48 gg,2,24 gggg,4,12 gggggg,6,8 c,1,48 cc,2,24 cccc,4,12 cccccccc,8,6 cccccccccccccccc,16,3"
  SET="g,1,16 gg,2,8 gc,2,8 cg,2,8, ggc,3,4 cgg,3,4 c,1,16 cc,2,8 ccc,3,4 cccc,4,4 ccccc,5,2 cccccc,6,2 ccccccc,7,2 cccccccc,8,2 ccccccccc,9,1 cccccccccc,10,1 ccccccccccc,11,1 cccccccccccc,12,1 ccccccccccccc,13,1 cccccccccccccc,14,1 cccccccccccccc,15,1 ccccccccccccccc,16,1"
else
  #SET="c,1,48 cc,2,24 cccc,4,12 cccccccc,8,6 cccccccccccccccc,16,3"
  #SET="c,1,80 cc,2,40 cccc,4,20 cccccccc,8,10 ccccccccccccccc,16,5"
  SET="c,1,16 cc,2,8 ccc,3,4 cccc,4,4 ccccc,5,2 cccccc,6,2 ccccccc,7,2 cccccccc,8,2 ccccccccc,9,1 cccccccccc,10,1 ccccccccccc,11,1 cccccccccccc,12,1 ccccccccccccc,13,1 cccccccccccccc,14,1 cccccccccccccc,15,1 ccccccccccccccc,16,1"
fi

pass=0
fail=0
failed_cases=""
for partition in 1 2 3 4 5 6 7 8 9 10; do
  CUTTYPE=

  if [ $partition -eq 1 ]; then
    CUTTYPE+=" -partition=cvc"
  elif [ $partition -eq 2 ]; then
    CUTTYPE+=" -partition=jcvc"
  elif [ $partition -eq 3 ]; then
    CUTTYPE+=" -partition=jbvc"
  elif [ $partition -eq 4 ]; then
    CUTTYPE+=" -partition=2dvc -balanceMasters=nodes"
  elif [ $partition -eq 5 ]; then
    CUTTYPE+=" -partition=hovc"
  elif [ $partition -eq 6 ]; then
    CUTTYPE+=" -partition=hivc"
  elif [ $partition -eq 7 ]; then
    CUTTYPE+=" -partition=oec"
  elif [ $partition -eq 8 ]; then
    CUTTYPE+=" -partition=iec"
  elif [ $partition -eq 9 ]; then
    CUTTYPE+=" -partition=od2vc"
  elif [ $partition -eq 10 ]; then
    CUTTYPE+=" -partition=od4vc"
  fi

  for task in $SET; do
    old_ifs=$IFS
    IFS=",";
    set $task;
    if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
      PFLAGS=" -pset=$1 -num_nodes=1"
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

    eval "sort -nu output_${hostname}_*.log -o output_${hostname}_0.log"
    eval "python $checker -t=1 $OUTPUT output_${hostname}_0.log &> .output_diff"

    cat .output_diff >> $LOG
    if ! grep -q "SUCCESS" .output_diff ; then
      let fail=fail+1
      if [ $partition -eq 1 ]; then
        failed_cases+="cartesian vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 2 ]; then
        failed_cases+="jagged cyclic vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 3 ]; then
        failed_cases+="jagged blocked vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 4 ]; then
        failed_cases+="2d checkerboard vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 5 ]; then
        failed_cases+="hybrid outgoing vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 6 ]; then
        failed_cases+="hybrid incoming vertex-cut $1 devices with $3 threads; "
      elif [ $partition -eq 7 ]; then
        failed_cases+="outgoing edge-cut $1 devices with $3 threads; "
      elif [ $partition -eq 8 ]; then
        failed_cases+="incoming edge-cut $1 devices with $3 threads; "
      elif [ $partition -eq 9 ]; then
        failed_cases+="over-decompose 2 cvc $1 devices with $3 threads; "
      elif [ $partition -eq 10 ]; then
        failed_cases+="over-decompose 4 cvc $1 devices with $3 threads; "
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
