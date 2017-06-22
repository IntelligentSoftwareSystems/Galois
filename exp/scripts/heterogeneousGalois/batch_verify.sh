#!/bin/sh

LOG=.verify_log

EXECS=( "bfs_pull-topological" "pagerank_pull-topological" "cc_pull-topological" "sssp_pull-topological" "bfs_push-filter" "pagerank_push-filter" "cc_push-filter" "sssp_push-filter" "bfs_push-worklist" "pagerank_push-worklist" "cc_push-worklist" "sssp_push-worklist" "bfs_push-topological" "pagerank_push-topological" "cc_push-topological" "sssp_push-topological" )

INPUTS=( "rmat25" "twitter-WWW10-component" )
INPUTS=( "rmat15" "rmat20" "rmat24" "road-USA" )

rm -f $LOG

current_dir=$(dirname "$0")
for input in "${INPUTS[@]}"
do
  for EXEC in "${EXECS[@]}"
  do
    $current_dir/verify.sh ${EXEC} ${input}
  done
done

