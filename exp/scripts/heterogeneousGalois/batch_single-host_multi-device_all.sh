#!/bin/sh

EXECS=("./bfs_push-topological_edge-cut" "./pagerank_push-topological_edge-cut" "./cc_push-topological_edge-cut" "./sssp_push-topological_edge-cut" "./bfs_pull-topological_edge-cut" "./pagerank_pull-topological_edge-cut" "./cc_pull-topological_edge-cut" "./sssp_pull-topological_edge-cut")
INPUTS=("/workspace/roshan/inputs/road/USA-road-d.USA.gr" "/workspace/roshan/inputs/scalefree/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.gr")
for i in "${EXECS[@]}"
do
  for j in "${INPUTS[@]}"
  do
    echo "./run_single-host_multi-device_all.sh ${i} ${j}"
    ./run_single-host_multi-device_all.sh ${i} ${j}
  done
done

