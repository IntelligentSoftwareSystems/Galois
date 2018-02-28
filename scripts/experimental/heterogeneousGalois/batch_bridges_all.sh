#!/bin/sh

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "cc_push" "cc_pull" "kcore_push" "kcore_pull" "pagerank_push" "pagerank_pull" "sssp_push" "sssp_pull" )
# fastest variants
EXECS=( "bfs_push" "cc_push" "pagerank_pull" "sssp_push" )

SET="1,02:00:00 2,01:30:00 4,01:00:00" #rmat28 gpu
SET="4,03:30:00 8,03:30:00 16,03:00:00 32,02:45:00 64,02:30:00 128,02:00:00" #clueweb12
SET="1,2:00:00 2,01:30:00 4,01:00:00 8,01:00:00 16,01:00:00 32,00:45:00 64,00:30:00 128,00:30:00" #rmat28
SETt="1,02:00:00" #twitter40 gpu
SETc="8,02:00:00 16,02:00:00" #clueweb12 gpu
SETk="8,02:00:00 16,02:00:00" #kron30 gpu
SETr="1,02:00:00 2,01:30:00 4,01:00:00 8,01:00:00 16,01:00:00" #rmat28 gpu

INPUTS=("twitter40;\"${SETt}\"")
INPUTS=("kron30;\"${SETk}\"")
INPUTS=("clueweb12;\"${SETc}\"")
INPUTS=("rmat28;\"${SETr}\"")

QUEUE=GPU-shared
QUEUE=RM
QUEUE=GPU
#HET=1

PARTS=( "cvc" "hovc" "2dvc" "iec" ) #rmat28/kron30
PARTS=( "cvc" "hivc" "2dvc" "oec" ) #clueweb12/twitter40
PARTS=( "cvc" )
PARTS=( "oec" ) #clueweb12/twitter40
PARTS=( "iec" ) #kron30/rmat28

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

