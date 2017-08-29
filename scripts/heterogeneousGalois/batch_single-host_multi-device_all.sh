#!/bin/sh

EXECS=( "bfs_pull-topological" "pagerank_pull-topological" "cc_pull-topological" "sssp_pull-topological" "bfs_push-worklist" "pagerank_push-worklist" "cc_push-worklist" "sssp_push-worklist" "bfs_push-filter" "pagerank_push-filter" "cc_push-filter" "sssp_push-filter" "bfs_push-topological" "pagerank_push-topological" "cc_push-topological" "sssp_push-topological" )

INPUTS=( "rmat25" "twitter-WWW10-component" "twitter-ICWSM10-component" "road-USA" )

for j in "${INPUTS[@]}"
do
  for i in "${EXECS[@]}"
  do
    echo "./run_single-host_multi-device_all.sh ${i} ${j}"
    ./run_single-host_multi-device_all.sh ${i} ${j}
  done
done

