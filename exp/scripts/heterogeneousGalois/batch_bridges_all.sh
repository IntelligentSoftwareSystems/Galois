#!/bin/sh

EXECS=( "bfs_pull-topological" "pagerank_pull-topological" "cc_pull-topological" "sssp_pull-topological" "bfs_push-filter" "pagerank_push-filter" "cc_push-filter" "sssp_push-filter" "bfs_push-worklist" "pagerank_push-worklist" "cc_push-worklist" "sssp_push-worklist" "bfs_push-topological" "pagerank_push-topological" "cc_push-topological" "sssp_push-topological" )
EXECS=( "bfs_push-filter_comm-updated-only" "pagerank_push-filter_comm-updated-only" "cc_push-filter_comm-updated-only" "sssp_push-filter_comm-updated-only" )

SET="1,c,4:00:00 2,cc,03:30:00 4,cccc,03:00:00 8,cccccccc,02:30:00 16,cccccccccccccccc,02:00:00 32,cccccccccccccccccccccccccccccccc,01:45:00 64,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,01:30:00 128,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,01:00:00"

INPUTS=("rmat20;\"${SET}\"")
INPUTS=("uk-2007;\"${SET}\"")
INPUTS=("twitter-ICWSM10-component;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")
INPUTS=("clueweb12;\"${SET}\"")

QUEUE=GPU-shared
QUEUE=GPU
QUEUE=RM

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
      echo "./run_bridges_all.sh ${i} ${1} ${2} $QUEUE"
      ./run_bridges_all.sh ${i} ${1} ${2} $QUEUE |& tee -a jobs
  done
done

