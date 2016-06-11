#!/bin/sh
# executes only on single machine
# assumes only one GPU device available

MPI=mpiexec
EXEC=$1
INPUT=$2
OUTPUT=$3
LOG=.verify_log

execname=$(basename "$EXEC" "")
inputdirname=$(dirname "$INPUT")
inputname=$(basename "$INPUT" ".gr")
if [[ $EXEC == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  inputname=${inputname}.transpose
  INPUT=${inputdirname}/${inputname}.gr
fi

if [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  ABELIAN_GALOIS_ROOT=/net/velocity/workspace/SourceCode/GaloisCpp
fi
checker=${ABELIAN_GALOIS_ROOT}/exp/scripts/result_checker.py

hostname=`hostname`

FLAGS=

SET=
if [[ $EXEC == *"vertex-cut"* ]]; then
  if [[ $INPUT == *"road"* ]]; then
    exit
  fi
  if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
    # assumes only one GPU device available
    SET="cc,2,2 gc,2,2 cg,2,2 cccc,4,1 gccc,4,1"
  else
    SET="cc,2,2 cccc,4,1"
  fi
else
  if [ -z "$ABELIAN_NON_HETEROGENEOUS" ]; then
    # assumes only one GPU device available
    SET="c,1,4 g,1,4 cc,2,2 gc,2,2 cg,2,2 cccc,4,1 gccc,4,1"
  else
    SET="c,1,4 cc,2,2 cccc,4,1"
  fi
fi

fail=0
failed_cases=""
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
  rm -f output_*.log
  eval "GALOIS_DO_NOT_BIND_THREADS=1 $MPI -n=$2 ${EXEC} -verify -runs=1 ${PFLAGS} -pset=$1 -t=$3 ${INPUT}" >>$LOG 2>&1
  outputs="output_${hostname}_0.log"
  i=1
  while [ $i -lt $2 ]; do
    outputs+=" output_${hostname}_${i}.log"
    let i=i+1
  done
  eval "python $checker $OUTPUT ${outputs} &> .output_diff"
  cat .output_diff >> $LOG
  if ! grep -q "SUCCESS" .output_diff ; then
    let fail=fail+1
    failed_cases+="$1 devices with $3 threads; "
  fi
  rm .output_diff
done

rm -f output_*.log

echo "---------------------------------------------------------------------------------------"
echo "Algorithm: " $execname
echo "Input: " $inputname
if [[ $fail == 0 ]] ; then
  echo "Status: SUCCESS"
else
  echo "Status: FAILED"
  echo $fail "failed test cases:" $failed_cases
fi
echo "---------------------------------------------------------------------------------------"

