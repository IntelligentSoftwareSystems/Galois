#!/bin/sh

# all benchmarks
EXECS=( "bfs_push" "bfs_pull" "cc_push" "cc_pull" "kcore_push" "kcore_pull" "pagerank_push" "pagerank_pull" "sssp_push" "sssp_pull" )
# fastest variants
EXECS=( "bfs_push" "cc_push" "pagerank_pull" "sssp_push" )

INPUTS=( "twitter40" "rmat26" "twitter50" "rmat28" "uk2007" )
INPUTS=( "twitter40" "rmat26" "rmat28" "uk2007" )
INPUTS=( "rmat28" "twitter40" )

for j in "${INPUTS[@]}"
do
  for i in "${EXECS[@]}"
  do
    echo "./run_single-host_multi-device_all.sh ${i} ${j}"
    ./run_single-host_multi-device_all.sh ${i} ${j}
    #echo "ABELIAN_VERIFY=1 ./run_single-host_multi-device_all.sh ${i} ${j}"
    #ABELIAN_VERIFY=1 ./run_single-host_multi-device_all.sh ${i} ${j}
  done
done

