#!/bin/sh

EXECS=( "bfs_pull-topological" "pagerank_pull-topological" "cc_pull-topological" "sssp_pull-topological" "bfs_push-filter" "pagerank_push-filter" "cc_push-filter" "sssp_push-filter" "bfs_push-worklist" "pagerank_push-worklist" "cc_push-worklist" "sssp_push-worklist" "bfs_push-topological" "pagerank_push-topological" "cc_push-topological" "sssp_push-topological" )
EXECS=( "bfs_push-filter_comm-updated-only" "pagerank_push-filter_comm-updated-only" "cc_push-filter_comm-updated-only" "sssp_push-filter_comm-updated-only" )

SET="1,2:00:00 2,01:30:00 4,01:00:00 8,00:45:00 16,00:30:00 32,00:20:00"

INPUTS=("clueweb12;\"${SET}\"")
INPUTS=("uk-2007;\"${SET}\"")
INPUTS=("rmat28;\"${SET}\"")
INPUTS=("twitter-ICWSM10-component;\"${SET}\"")

QUEUE=gpu
QUEUE=normal
QUEUE=development

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
      echo "./run_stampede_all.sh ${i} ${1} ${2} $QUEUE"
      ./run_stampede_all.sh ${i} ${1} ${2} $QUEUE |& tee -a jobs
  done
done

