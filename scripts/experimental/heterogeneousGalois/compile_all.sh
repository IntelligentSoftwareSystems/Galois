#!/bin/sh

if ! [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  BASE_DIR=${ABELIAN_GALOIS_ROOT}/dist_apps/experimental
else
  BASE_DIR=..
fi
INPUT_DIR=${BASE_DIR}/compiler_inputs
OUTPUT_DIR=${BASE_DIR}/compiler_outputs

if [ -n "$1" ]; then
  threads=$1
else
  threads=1
fi
count=0
for input in $INPUT_DIR/*.cpp; do
  name=$(basename "$input" ".cpp")
  ./compile.sh $input ${OUTPUT_DIR}/$name &
  count=$((count+1))
  if [[ $count == $threads ]]; then
    wait
    count=0
  fi
done
if [[ $count != 0 ]]; then
  wait
fi

