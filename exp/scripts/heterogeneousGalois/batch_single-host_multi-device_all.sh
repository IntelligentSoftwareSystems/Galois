#!/bin/sh

EXECS=( "bfs_pull-topological_edge-cut" "pagerank_pull-topological_edge-cut" "cc_pull-topological_edge-cut" "sssp_pull-topological_edge-cut" "bfs_push-worklist_edge-cut" "pagerank_push-worklist_edge-cut" "cc_push-worklist_edge-cut" "sssp_push-worklist_edge-cut" "bfs_push-filter_edge-cut" "pagerank_push-filter_edge-cut" "cc_push-filter_edge-cut" "sssp_push-filter_edge-cut" "bfs_push-topological_edge-cut" "pagerank_push-topological_edge-cut" "cc_push-topological_edge-cut" "sssp_push-topological_edge-cut" )

INPUTS=( "rmat25" "twitter-WWW10-component" "twitter-ICWSM10-component" "road-USA" )

for j in "${INPUTS[@]}"
do
  for i in "${EXECS[@]}"
  do
    if [ -n "$1" ]; then
      echo "./run_vtune_single-host_multi-device_all.sh ${i} ${j}"
      ./run_vtune_single-host_multi-device_all.sh ${i} ${j}
    else
      echo "./run_single-host_multi-device_all.sh ${i} ${j}"
      ./run_single-host_multi-device_all.sh ${i} ${j}
    fi
  done
done

