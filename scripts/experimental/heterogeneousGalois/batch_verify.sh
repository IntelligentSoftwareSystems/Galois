#!/bin/sh

LOG=.verify_log

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "kcore_push" "kcore_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" "pagerank_push" "pagerank_pull" )
EXECS=( "bfs_push" "kcore_push" "cc_push" "sssp_push" "pagerank_pull" )

INPUTS=( "rmat25" "twitter-WWW10-component" )
INPUTS=( "rmat15" "rmat20" "rmat24" "road-USA" )
INPUTS=( "rmat15" "rmat20" )
INPUTS=( "rmat20" )
INPUTS=( "rmat15" )

rm -f $LOG

current_dir=$(dirname "$0")
for input in "${INPUTS[@]}"
do
  for EXEC in "${EXECS[@]}"
  do
    $current_dir/verify.sh ${EXEC} ${input} "--exec=Sync"
    $current_dir/verify.sh ${EXEC} ${input} "--exec=Async"
    rm -f $LOG
  done
done

