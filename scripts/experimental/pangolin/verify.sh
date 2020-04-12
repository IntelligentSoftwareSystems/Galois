#!/bin/bash
# Usage: ./verify.sh <ABELIAN_EXECUTABLE_NAME> <INPUT_GRAPH_NAME>

execname=$1
inputname=$2
option=$3
bin=${execname}
if [[ $execname == *"tc"* ]]; then
  bin="tc_mine"
fi
execdirname="./${execname}"
NTHREADS="56"
MINSUP="500"
EXEC=${execdirname}/${bin}
inputdirname=/net/ohm/export/iss/inputs/Mining
outputdirname=/net/ohm/export/iss/pangolin-outputs

filetype=gr
extension=csgr
if [[ $execname == *"fsm"* ]]; then
  filetype=adj
  extension=sadj
fi

IFS='_' read -ra EXECP <<< "$execname"
problem=${EXECP[0]}

SIZES="3"
if [[ $execname == *"fsm"* ]]; then
  SIZES="2"
fi

if [[ $execname == *"kcl"* ]]; then
  SIZES="4 5"
fi

if [[ $execname == *"motif"* ]]; then
  SIZES="3 4"
fi

FLAGS=
if [[ $execname == *"fsm"* ]]; then
  FLAGS="-ms=$MINSUP"
fi

#FLAGS+=" -t=56"
OUTPUT=${outputdirname}/${inputname}.${problem}.$K
INPUT=${inputdirname}/${inputname}.${extension}
checker=${outputdirname}/result_checker.py
pass=0
fail=0
failed_cases=""
check_output="my-output.log"


for K in $SIZES; do
	for NT in $NTHREADS; do
		LOG=${execname}-${inputname}-$K-$NT.log
		echo "${EXEC} $filetype ${INPUT} -t=$NT -k=$K $FLAGS -v > $LOG"
		eval "${EXEC} $filetype ${INPUT} -t=$NT -k=$K $FLAGS -v" > $LOG 2>> error.log
		echo "python $checker ${execname} ${inputname} $K $MINSUP $LOG &> ${check_output}"
		eval "python $checker ${execname} ${inputname} $K $MINSUP $LOG &> ${check_output}"
		#cat ${check_output}
		if ! grep -q "SUCCESS" ${check_output} ; then
			let fail=fail+1
			failed_cases+="${execname} ${inputname} k=$K t=$NT"
		else
			let pass=pass+1
		fi
		rm -f ${check_output}
	done
done


echo "---------------------------------------------------------------------------------------"
echo "Algorithm: " $execname
echo "Input: " $inputname
echo "Runtime option: " $option
echo $pass "passed test cases"
if [[ $fail == 0 ]] ; then
  echo "Status: SUCCESS"
else
  echo $fail "failed test cases:" $failed_cases
  echo "Status: FAILED"
fi
echo "---------------------------------------------------------------------------------------"

