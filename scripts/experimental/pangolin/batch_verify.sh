#!/bin/sh

# all benchmarks
EXECS=( "tc" "kcl" "motif" "fsm" )
#EXECS=( "tc" "kcl" "fsm" )

#INPUTS=( "mico" "patent" "youtube" )
INPUTS=( "citeseer" )
#INPUTS=( "livej" "orkut" )

current_dir=$(dirname "$0")
for input in "${INPUTS[@]}"; do
  for EXEC in "${EXECS[@]}"; do
    $current_dir/verify.sh ${EXEC} ${input}
  done
done

