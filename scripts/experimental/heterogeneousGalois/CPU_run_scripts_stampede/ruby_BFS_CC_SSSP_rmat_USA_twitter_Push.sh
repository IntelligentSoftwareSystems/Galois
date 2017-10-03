#!/bin/bash
benchmark=$2
variant=$3
partition=$4
comm_mode=${11}

algo=$benchmark"_push-"$variant"_edge-cut"
echo $algo

./$algo "-help"

ENV_options="MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1"


GRAPH_twitter="/scratch/03279/roshand/dist-inputs/twitter-ICWSM10-component.gr"
GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/rmat28.gr" #Randomized rmat28

cmd_options_reset="-maxIterations=10000 -verify=0 -t=$1 -comm_mode=${comm_mode}"
cmd_options="-maxIterations=10000 -verify=0 -t=$1 -comm_mode=${comm_mode}"

if [ $benchmark = "pagerank" ]; then
	cmd_options=$cmd_options"  -tolerance=0.0000001"
fi

if [ $benchmark = "cc" ]; then
  GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/symmetric/rmat28.sgr"
  GRAPH_twitter="/scratch/03279/roshand/dist-inputs/symmetric/twitter-ICWSM10-component.sgr"
fi


#RMAT25
if [ $5 = "rmat28" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options" -srcNodeId=155526494"
  fi
ruby ../../../../../../Distributed_latest/scripts/stampede_jobs.rb  -t "01:42:30" -q "normal" -n 4 -N 4 -i dist_run_script_generated -o ./LOG_jul_31/LOG_${algo}_TH$1\_CM${comm_mode}\_rmat28.rgr_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_rmat28 $cmd_options " -s $6  -e $7 -k 2
fi

#  twitter ICWSM ##sssp
if [ $8 = "twitter" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options_reset
    cmd_options=$cmd_options" -srcNodeId=33643219"
  fi
ruby ../../../../../../Distributed_latest/scripts/stampede_jobs.rb  -t "01:48:00" -q "normal" -n 4 -N 4 -i dist_run_script_generated -o ./LOG_jul_31/LOG_${algo}_TH$1\_CM${comm_mode}\_Twitter-ICWSM10_  -A "Galois"  -c "$ENV_options ibrun ./$algo $GRAPH_twitter $cmd_options" -s $9  -e ${10}  -k 2
fi
