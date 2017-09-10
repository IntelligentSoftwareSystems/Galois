#!/bin/bash
benchmark=$2
variant=$3
partition=$4
comm_mode="0"

algo=$benchmark"_pull-"$variant
echo $algo

./$algo "-help"

if [ $partition != "0" ]; then
  echo "Parition should be 0 since it's edgeCut : Exiting"
  exit
fi

ENV_options="MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1"

GRAPH_rmat="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/scalefree/NEW/transpose/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.transpose.gr"
GRAPH_usa="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/road/USA-road-d.USA-trans.gr"

GRAPH_twitter="/scratch/03279/roshand/dist-inputs/transpose/twitter-ICWSM10-component.tgr"   #"/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/withRandomWeights/transpose/twitter-ICWSM10-component_withRandomWeights.transpose.gr"
GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/transpose/rmat28.tgr" #Randomized rmat28
GRAPH_rmat15="/scratch/03279/roshand/dist-inputs/transpose/rmat15.tgr" #Randomized rmat28

cmd_options_reset="-maxIterations=10000 -verify=0 -t=$1 -enableVertexCut=${partition}"
cmd_options="-maxIterations=10000 -verify=0 -t=$1 -enableVertexCut=${partition}"

if [ $benchmark = "pagerank" ]; then
	cmd_options=$cmd_options"  -tolerance=0.0000001"
fi

if [ $benchmark = "cc" ]; then
  GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/symmetric/rmat28.sgr"
  GRAPH_twitter="/scratch/03279/roshand/dist-inputs/symmetric/twitter-ICWSM10-component.sgr"
fi


#RMAT28
if [ $5 = "rmat28" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options" -srcNodeId=155526494"
  fi
ruby /work/02982/ggill0/Distributed_latest/scripts/stampede_jobs.rb  -t "01:42:30" -q "normal" -n 4 -N 4 -i dist_run_script_generated -o ./LOG_RUNS/LOG_${algo}_TH$1\_ECUT\_${partition}\_rmat28.tgr_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_rmat28 $cmd_options " -s $6  -e $7 -k 2
fi

#  twitter ICWSM ##sssp
if [ $8 = "twitter" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options_reset
    cmd_options=$cmd_options" -srcNodeId=33643219"
  fi
ruby /work/02982/ggill0/Distributed_latest/scripts/stampede_jobs.rb  -t "01:48:00" -q "normal" -n 4 -N 4 -i dist_run_script_generated -o ./LOG_RUNS/LOG_${algo}_TH$1\_ECUT\_${partition}\_Twitter-ICWSM10.tgr_  -A "Galois"  -c "$ENV_options ibrun ./$algo $GRAPH_twitter $cmd_options" -s $9  -e ${10}  -k 2
fi

#RMAT15
if [[ ${11} = "rmat15" ]]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options" -srcNodeId=0"
  fi
ruby /work/02982/ggill0/Distributed_latest/scripts/stampede_jobs.rb  -t "00:12:30" -q "development" -n 4 -N 4 -i dist_run_script_generated -o ./DEV_RUNS/DEV_LOG_${algo}_TH$1\_ECUT\_${partition}\_rmat15.tgr_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_rmat15 $cmd_options " -s ${12}  -e ${13} -k 2
fi


