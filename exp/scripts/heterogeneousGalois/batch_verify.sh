#!/bin/sh

LOG=.verify_log

EXECS=("./bfs_push-topological_edge-cut" "./pagerank_push-topological_edge-cut" "./cc_push-topological_edge-cut" "./sssp_push-topological_edge-cut" "./bfs_pull-topological_edge-cut" "./pagerank_pull-topological_edge-cut" "./cc_pull-topological_edge-cut" "./sssp_pull-topological_edge-cut" "./bfs_push-worklist_edge-cut" "./pagerank_push-worklist_edge-cut" "./cc_push-worklist_edge-cut" "./sssp_push-worklist_edge-cut" "./bfs_push-filter_edge-cut" "./pagerank_push-filter_edge-cut" "./cc_push-filter_edge-cut" "./sssp_push-filter_edge-cut" )
INPUTS=("/workspace/roshan/inputs/scalefree/rmat16-2e20-a=0.57-b=0.19-c=0.19-d=.05.gr" "/workspace/roshan/inputs/road/USA-road-d.USA.gr" "/workspace/roshan/inputs/scalefree/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.gr")

rm -f $LOG

for algo in "${EXECS[@]}"
do
  for input in "${INPUTS[@]}"
  do
    algoname=$(basename "$algo" "")
    IFS='_' read -ra ALGO <<< "$algoname"
    problem=${ALGO[0]}
    output=${input//\/inputs\//\/outputs\/}
    output=${output//\.gr/\.${problem}}
    ./verify.sh ${algo} ${input} ${output}
  done
done

