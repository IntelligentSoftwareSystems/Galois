#!/bin/sh

LOG=.verify_log

EXECS=( "bfs_pull-topological_edge-cut" "pagerank_pull-topological_edge-cut" "cc_pull-topological_edge-cut" "sssp_pull-topological_edge-cut" "bfs_push-worklist_edge-cut" "pagerank_push-worklist_edge-cut" "cc_push-worklist_edge-cut" "sssp_push-worklist_edge-cut" "bfs_push-filter_edge-cut" "pagerank_push-filter_edge-cut" "cc_push-filter_edge-cut" "sssp_push-filter_edge-cut" "bfs_push-topological_edge-cut" "pagerank_push-topological_edge-cut" "cc_push-topological_edge-cut" "sssp_push-topological_edge-cut" )

INPUTS=( "rmat15" "rmat20" "rmat24" "rmat25" "twitter-WWW10-component" "twitter-ICWSM10-component" "road-USA" )

rm -f $LOG

for input in "${INPUTS[@]}"
do
  for EXEC in "${EXECS[@]}"
  do
    ./verify.sh ${EXEC} ${input}
  done
done

