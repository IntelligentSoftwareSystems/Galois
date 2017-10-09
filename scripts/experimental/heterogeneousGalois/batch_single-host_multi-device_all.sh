#!/bin/sh

# fastest variants
EXECS=( "bfs_push" "pagerank_pull" "kcore_push" "cc_push" "sssp_push" )
# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "kcore_push" "kcore_pull" "cc_push" "cc_pull" "sssp_push" "sssp_pull" )

INPUTS=( "rmat25" "twitter-WWW10-component" "twitter-ICWSM10-component" "road-USA" )

for j in "${INPUTS[@]}"
do
  for i in "${EXECS[@]}"
  do
    echo "./run_single-host_multi-device_all.sh ${i} ${j}"
    ./run_single-host_multi-device_all.sh ${i} ${j}
  done
done

