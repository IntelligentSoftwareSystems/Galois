#!/bin/sh

SET1="1,2,cg,2:00:00 2,4,cgcg,01:30:00 4,8,cgcgcgcg,01:00:00 8,16,cgcgcgcgcgcgcgcg,00:45:00 16,32,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:30:00 32,64,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:20:00" # push-worklist rmat28 twitter-ICWSM(except pacgerank)
SET2="1,2,cg,2:00:00 2,4,cgcg,01:30:00 4,8,cgcgcgcg,01:00:00 8,16,cgcgcgcgcgcgcgcg,01:00:00 16,32,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:45:00 32,64,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:45:00" # push-worklist twitter-ICWSM for pacgerank
SET3="1,1,g,2:00:00 2,2,gg,01:30:00 4,4,gggg,01:00:00 8,8,gggggggg,00:45:00 16,16,gggggggggggggggg,00:30:00 32,32,gggggggggggggggggggggggggggggggg,00:20:00" # push-worklist rmat28 twitter-ICWSM(except pagerank)
SET4="1,1,g,2:00:00 2,2,gg,01:30:00 4,4,gggg,01:00:00 8,8,gggggggg,01:00:00 16,16,gggggggggggggggg,00:45:00 32,32,gggggggggggggggggggggggggggggggg,00:45:00" # push-worklist twitter-ICWSM for pagerank
SET5="32,32,gggggggggggggggggggggggggggggggg,01:30:00" 
SET6="32,64,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,02:30:00"
SET7="8,16,cgcgcgcgcgcgcgcg,00:45:00 16,32,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:30:00 32,64,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,00:20:00" # push-worklist rmat28 twitter-ICWSM(except pacgerank)
SET8="16,32,cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcg,02:00:00"
SET9="8,16,cgcgcgcgcgcgcgcg,01:45:00"
SET9="8,8,gggggggg,00:45:00 16,16,gggggggggggggggg,00:30:00" 

EXECS=( "bfs_pull-topological_edge-cut" "pagerank_pull-topological_edge-cut" "cc_pull-topological_edge-cut" "sssp_pull-topological_edge-cut" "bfs_push-worklist_edge-cut" "pagerank_push-worklist_edge-cut" "cc_push-worklist_edge-cut" "sssp_push-worklist_edge-cut" "bfs_push-filter_edge-cut" "pagerank_push-filter_edge-cut" "cc_push-filter_edge-cut" "sssp_push-filter_edge-cut" "bfs_push-topological_edge-cut" "pagerank_push-topological_edge-cut" "cc_push-topological_edge-cut" "sssp_push-topological_edge-cut" )

INPUTS=("rmat28;\"${SET1}\"" "twitter-ICWSM10-component;\"${SET1}\"")

for j in "${INPUTS[@]}"
do
  IFS=";";
  set $j;
  for i in "${EXECS[@]}"
  do
      echo "./run_multi-host_multi-device_all.sh ${i} ${1} ${2}"
      ./run_multi-host_multi-device_all.sh ${i} ${1} ${2} |& tee -a jobs
  done
done

