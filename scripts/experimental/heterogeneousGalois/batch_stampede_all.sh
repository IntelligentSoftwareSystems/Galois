#!/bin/sh

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "cc_push" "cc_pull" "kcore_push" "kcore_pull" "pagerank_push" "pagerank_pull" "sssp_push" "sssp_pull" )
# fastest variants
EXECS=( "bfs_push" "cc_push" "pagerank_pull" "sssp_push" )

SET="1,2:00:00 2,01:30:00 4,01:00:00 8,00:45:00 16,00:30:00 32,00:20:00"
SET="128,00:30:00 64,00:45:00 32,01:00:00"
SET="64,01:00:00 32,01:30:00 16,02:00:00" 
SETc="256,01:00:00 128,01:00:00 64,01:15:00 32,01:30:00 16,01:45:00"
SETk="256,01:00:00 128,01:00:00 64,01:15:00 32,01:30:00 16,01:45:00 8,02:00:00 4,02:30:00"
SETr="256,01:00:00 128,01:00:00 64,01:15:00 32,01:30:00 16,01:45:00 8,02:00:00 4,02:30:00 2,02:30:00 1,02:30:00"

INPUTS=("twitter40;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")
INPUTS=("kron30;\"${SET}\"")
INPUTS=("clueweb12;\"${SET}\"")
INPUTS=("wdc12;\"${SET}\"")
INPUTS=("rmat28;\"${SETr}\"" "kron30;\"${SETk}\"" "clueweb12;\"${SETc}\"")

QUEUE=development
QUEUE=normal

PARTS=( "cvc" "hivc" "2dvc" "oec" ) #clueweb12
PARTS=( "cvc" "hovc" "2dvc" "iec" ) #rmat28/kron30
PARTS=( "cvc" )

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

