#!/bin/sh

LOG=.verify_log

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" "pagerank_push" "pagerank_pull" "kcore_push" "kcore_pull" )
EXECS=( "bfs_push_async" )
EXECS=( "bfs_push_async" "bfs_pull_async" "cc_push_async" "cc_pull_async" "sssp_push_async" "sssp_pull_async" "pagerank_push_async" "pagerank_pull_async" "kcore_push_async" "kcore_pull_async" )
EXECS=( "bfs_push_async" "bfs_pull_async" "cc_push_async" "cc_pull_async" "sssp_push_async" "sssp_pull_async" "pagerank_push_async" "kcore_push_async" "kcore_pull_async" )
EXECS=( "pagerank_pull_async" )
EXECS=( "pagerank_pull" )

INPUTS=( "rmat25" "twitter-WWW10-component" )
INPUTS=( "rmat15" "rmat20" "rmat24" "road-USA" )
INPUTS=( "rmat15" "rmat20" )

rm -f $LOG

current_dir=$(dirname "$0")
for input in "${INPUTS[@]}"
do
  for EXEC in "${EXECS[@]}"
  do
    $current_dir/verify.sh ${EXEC} ${input}
  done
done

