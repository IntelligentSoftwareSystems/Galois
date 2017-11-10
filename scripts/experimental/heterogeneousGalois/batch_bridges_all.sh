#!/bin/sh

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "pagerank_pull" "pagerank_push" "kcore_push" "kcore_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" )
# fastest variants
EXECS=( "bfs_push" "pagerank_pull" "kcore_push" "cc_push" "sssp_push" )

SET="1,02:00:00 2,01:30:00 4,01:00:00" #rmat28 gpu
SET="4,03:30:00 8,03:30:00 16,03:00:00 32,02:45:00 64,02:30:00 128,02:00:00" #clueweb12
SET="1,2:00:00 2,01:30:00 4,01:00:00 8,01:00:00 16,01:00:00 32,00:45:00 64,00:30:00 128,00:30:00" #rmat28

INPUTS=("twitter-ICWSM10-component;\"${SET}\"")
INPUTS=("twitter-WWW10-component;\"${SET}\"")
INPUTS=("rmat30;\"${SET}\"")
INPUTS=("uk-2007;\"${SET}\"")
INPUTS=("kron30;\"${SET}\"")
INPUTS=("clueweb12;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")

QUEUE=GPU-shared
QUEUE=RM
QUEUE=GPU
#HET=1

PARTS=( "cvc" "hovc" "2dvc" "iec" ) #rmat30
PARTS=( "cvc" "hivc" "2dvc" "oec" ) #clueweb12
PARTS=( "oec" ) #clueweb12 pr
PARTS=( "iec" ) #kron30 or rmat28 pr
PARTS=( "cvc" ) #default

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
    for p in "${PARTS[@]}"
    do
      echo "./run_bridges_all.sh ${i} ${1} ${2} $QUEUE $p $HET"
      ./run_bridges_all.sh ${i} ${1} ${2} $QUEUE $p $HET |& tee -a jobs
    done
  done
done

