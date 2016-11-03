#!/bin/sh

EXECS=( "bfs_pull-topological" "pagerank_pull-topological" "cc_pull-topological" "sssp_pull-topological" "bfs_push-worklist" "pagerank_push-worklist" "cc_push-worklist" "sssp_push-worklist" "bfs_push-filter" "pagerank_push-filter" "cc_push-filter" "sssp_push-filter" "bfs_push-topological" "pagerank_push-topological" "cc_push-topological" "sssp_push-topological" )

INPUTS=("twitter-ICWSM10-component;\"1,c,2:00:00\"" "twitter-WWW10-component;\"1,c,2:00:00\"" "rmat28;\"1,c,2:30:00\"" )
INPUTS=("rmat28;\"1,gggg,2:00:00 2,gggggggg,01:30:00 4,gggggggggggggggg,01:15:00 8,gggggggggggggggggggggggggggggggg,01:00:00 16,gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg,00:45:00\"")
INPUTS=("twitter-ICWSM10-component;\"1,gggg,2:00:00 2,gggggggg,01:30:00 4,gggggggggggggggg,01:15:00 8,gggggggggggggggggggggggggggggggg,01:00:00 16,gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg,00:45:00\"")
INPUTS=("rmat25;\"1,g,2:00:00 2,gg,01:30:00 3,ggg,1:15:00 4,gggg,01:00:00\"" "twitter-WWW10-component;\"1,g,2:00:00 2,gg,01:30:00 3,ggg,1:15:00 4,gggg,01:00:00\"")
INPUTS=("rmat28;\"1,g,2:30:00 2,gg,02:00:00 3,ggg,1:45:00 4,gggg,01:30:00\"" "twitter-ICWSM10-component;\"1,g,2:00:00 2,gg,01:30:00 3,ggg,1:15:00 4,gggg,01:00:00\"")
INPUTS=("rmat25;\"2,cg,01:45:00 3,cgg,1:30:00 4,cggg,01:15:00 5,cgggg,1:00:00 \"")

QUEUE=RM
QUEUE=GPU
QUEUE=GPU-shared

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

