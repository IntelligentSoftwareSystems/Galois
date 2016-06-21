#!/bin/sh

if ! [ -z "$ABELIAN_GALOIS_ROOT" ]; then
  BASE_DIR=${ABELIAN_GALOIS_ROOT}/exp/apps
else
  BASE_DIR=..
fi
INPUT_DIR=${BASE_DIR}/compiler_inputs
OUTPUT_DIR=${BASE_DIR}/compiler_outputs

for input in $INPUT_DIR/*.cpp; do
  name=$(basename "$input" ".cpp")
  ./compile.sh $input ${OUTPUT_DIR}/$name
done

