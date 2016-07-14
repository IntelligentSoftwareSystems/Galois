#!/bin/sh

EXECS=("./bfs_push-topological_edge-cut" "./pagerank_push-topological_edge-cut" "./cc_push-topological_edge-cut" "./sssp_push-topological_edge-cut" "./bfs_pull-topological_edge-cut" "./pagerank_pull-topological_edge-cut" "./cc_pull-topological_edge-cut" "./sssp_pull-topological_edge-cut" "./bfs_push-worklist_edge-cut" "./pagerank_push-worklist_edge-cut" "./cc_push-worklist_edge-cut" "./sssp_push-worklist_edge-cut" "./bfs_push-filter_edge-cut" "./pagerank_push-filter_edge-cut" "./cc_push-filter_edge-cut" "./sssp_push-filter_edge-cut" )
INPUTS=("/workspace/dist-inputs/road-USA.gr" "/workspace/dist-inputs/rmat24.gr" "/workspace/dist-inputs/rmat25.gr" )
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

