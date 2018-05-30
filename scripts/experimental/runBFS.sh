#!/bin/bash

declare -A inputsMap

inputsMap["r4"]="/net/ohm/export/iss/inputs/random/r4-2e26.gr"
inputsMap["rmat"]="/net/ohm/export/iss/inputs/scalefree/rmat16-2e26-a=0.57-b=0.19-c=0.19-d=.05.gr"
inputsMap["twitter"]="/net/ohm/export/iss/inputs/unweighted/twitter-WWW10-component.gr"


serialAlgos="SerialSync Serial"
serialRep="`seq 1 3`"
tag=${tag="tag"}

for algo in $serialAlgos; do 
  for input in "${!inputsMap[@]}"; do 
    for i in 1 2 3; do 
      ./lonestar/bfs/bfs -algo=${algo}  "${inputsMap[$input]}" -noverify  ; 
    done 2>&1 | tee bfs-${tag}-${algo}-${input}.log 
  done
done

parallelAlgos="Async Sync Sync2p"
threads="1 `seq 5 5 40`"

for algo in $parallelAlgos; do
  for input in "${!inputsMap[@]}"; do 
    for t in $threads; do
      ./lonestar/bfs/bfs -algo=${algo}  "${inputsMap[$input]}" -noverify  -t $t; 
    done 2>&1 | tee bfs-${tag}-${algo}-${input}.log 
  done
done

