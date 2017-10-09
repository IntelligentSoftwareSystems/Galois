#!/bin/sh

# fastest variants
EXECS=( "bfs_push" "pagerank_pull" "kcore_push" "cc_push" "sssp_push" )
# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "kcore_push" "kcore_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" )

SET="1,2:00:00 2,01:30:00 4,01:00:00 8,00:45:00 16,00:30:00 32,00:20:00"
SET="128,00:30:00 64,00:45:00 32,01:00:00"
SET="64,01:00:00 32,01:30:00 16,02:00:00" 

INPUTS=("uk-2007;\"${SET}\"")
INPUTS=("twitter-ICWSM10-component;\"${SET}\"")
INPUTS=("rmat24;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")
INPUTS=("clueweb12;\"${SET}\"")
INPUTS=("rmat30;\"${SET}\"")

QUEUE=gpu
QUEUE=development
QUEUE=normal

PARTS=( "cvc" "hivc" "2dvc" "oec" ) #clueweb12
PARTS=( "cvc" "hovc" "2dvc" "iec" ) #rmat30

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
    for p in "${PARTS[@]}"
    do
      echo "./run_stampede_all.sh ${i} ${1} ${2} $QUEUE $p"
      ./run_stampede_all.sh ${i} ${1} ${2} $QUEUE $p |& tee -a jobs
    done
  done
done

