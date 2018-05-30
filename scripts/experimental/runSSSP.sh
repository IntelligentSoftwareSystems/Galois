#!/bin/bash

declare -A inputsMap

inputsMap["r4"]="/net/ohm/export/iss/inputs/random/r4-2e26.gr"
inputsMap["rmat"]="/net/ohm/export/iss/inputs/scalefree/rmat16-2e26-a=0.57-b=0.19-c=0.19-d=.05.gr"
inputsMap["road"]="/net/ohm/export/iss/inputs/road/osm-eur-karlsruhe.gr"


app="./lonestar/sssp/sssp"
# serialAlgos="dijkstra serDelta serDeltaTiled"
serialAlgos="serDelta serDeltaTiled"
serialRep="`seq 1 3`"
tag=${tag="tag"}

for algo in $serialAlgos; do 
  for input in "${!inputsMap[@]}"; do 
    for i in 1 2 3; do 
      ${app} -algo=${algo}  "${inputsMap[$input]}" -noverify  ; 
    done 2>&1 | tee $(basename ${app})-${tag}-${algo}-${input}.log 
  done
done

parallelAlgos="deltaStep deltaTiled"
threads="1 `seq 5 5 40`"
# threads="40"

for algo in $parallelAlgos; do
  for input in "${!inputsMap[@]}"; do 
    for t in $threads; do
      ${app} -algo=${algo}  "${inputsMap[$input]}" -noverify  -t $t; 
    done 2>&1 | tee $(basename ${app})-${tag}-${algo}-${input}.log 
  done
done

