#!/bin/sh

INPUT_DIR=../compiler_inputs

for input in $INPUT_DIR/*.cpp; do
  name=$(basename "$input" ".cpp")
  ./compiler.sh $input $name
done

