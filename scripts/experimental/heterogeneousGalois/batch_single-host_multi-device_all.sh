#!/bin/sh

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "cc_push" "cc_pull" "kcore_push" "kcore_pull" "pagerank_push" "pagerank_pull" "sssp_push" "sssp_pull" )
# fastest variants
EXECS=( "bfs_push" "cc_push" "pagerank_pull" "sssp_push" )
EXECS=( "bfs_push" "bfs_push_async" )
EXECS=( "bfs_push" "bfs_push_async" "cc_push" "cc_push_async" "sssp_push" "sssp_push_async" "pagerank_pull" "pagerank_pull_async" )
EXECS=( "pagerank_pull" "pagerank_pull_async" )
EXECS=( "pagerank_pull_async" )

INPUTS=( "twitter40" "rmat26" "twitter50" "rmat28" "uk2007" )
INPUTS=( "twitter40" "rmat26" "rmat28" "uk2007" )
INPUTS=( "twitter40" )
INPUTS=( "rmat28" )
INPUTS=( "twitter40" "rmat28" )

export ABELIAN_VERTEX_CUT=1
export ABELIAN_VERIFY=1

for j in "${INPUTS[@]}"
do
  for i in "${EXECS[@]}"
  do
    echo "./run_single-host_multi-device_all.sh ${i} ${j}"
    ./run_single-host_multi-device_all.sh ${i} ${j}
  done
done

