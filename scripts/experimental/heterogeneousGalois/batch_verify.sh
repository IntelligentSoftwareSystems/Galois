#!/bin/sh

LOG=.verify_log

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "kcore_push" "kcore_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" "pagerank_push" "pagerank_pull" )
EXECS=( "bfs_push" "kcore_push" "cc_push" "sssp_push" "pagerank_pull" )
EXECS=( "bfs_push" "bfs_push_async" )
EXECS=( "bfs_push" "kcore_push" "cc_push" "sssp_push" "pagerank_pull" "bfs_push_async" "kcore_push_async" "cc_push_async" "sssp_push_async" "pagerank_pull_async" )

INPUTS=( "rmat25" "twitter-WWW10-component" )
INPUTS=( "rmat15" "rmat20" "rmat24" "road-USA" )
INPUTS=( "rmat20" )
INPUTS=( "rmat15" "rmat20" )
INPUTS=( "rmat15" )

rm -f $LOG

current_dir=$(dirname "$0")
for input in "${INPUTS[@]}"
do
  for EXEC in "${EXECS[@]}"
  do
    $current_dir/verify.sh ${EXEC} ${input}
    rm -f $LOG
  done
done

