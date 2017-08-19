#!/bin/sh

EXECS=( "bfs_push-filter" "pagerank_pull-topological" "kcore_push-filter" "cc_push-filter" "sssp_push-filter" )

SET="1,2:00:00 2,01:30:00 4,01:00:00 8,00:45:00 16,00:30:00 32,00:20:00"
SET="128,00:30:00 64,00:45:00 32,01:00:00"

INPUTS=("uk-2007;\"${SET}\"")
INPUTS=("twitter-ICWSM10-component;\"${SET}\"")
INPUTS=("rmat24;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")
INPUTS=("clueweb12;\"${SET}\"")

QUEUE=gpu
QUEUE=development
QUEUE=normal

PART=oec
PART=hivc
PART=2dvc
PART=cvc

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
      echo "./run_stampede_all.sh ${i} ${1} ${2} $QUEUE $PART"
      ./run_stampede_all.sh ${i} ${1} ${2} $QUEUE $PART |& tee -a jobs
  done
done

