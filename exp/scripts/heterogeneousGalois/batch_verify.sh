#!/bin/sh

LOG=.verify_log

EXECS=("./bfs_push-topological_edge-cut" "./pagerank_push-topological_edge-cut" "./cc_push-topological_edge-cut" "./sssp_push-topological_edge-cut" "./bfs_pull-topological_edge-cut" "./pagerank_pull-topological_edge-cut" "./cc_pull-topological_edge-cut" "./sssp_pull-topological_edge-cut" "./bfs_push-worklist_edge-cut" "./pagerank_push-worklist_edge-cut" "./cc_push-worklist_edge-cut" "./sssp_push-worklist_edge-cut" "./bfs_push-filter_edge-cut" "./pagerank_push-filter_edge-cut" "./cc_push-filter_edge-cut" "./sssp_push-filter_edge-cut" )
INPUTS=("/workspace/dist-inputs/rmat15.gr" "/workspace/dist-inputs/rmat20.gr" "/workspace/dist-inputs/rmat24.gr" "/workspace/dist-inputs/rmat25.gr" "/workspace/dist-inputs/road-USA.gr" )

rm -f $LOG

for input in "${INPUTS[@]}"
do
  for algo in "${EXECS[@]}"
  do
    algoname=$(basename "$algo" "")
    IFS='_' read -ra ALGO <<< "$algoname"
    problem=${ALGO[0]}
    output=${input//inputs/outputs}
    output=${output//\.gr/\.${problem}}
    ./verify.sh ${algo} ${input} ${output}
  done
done

