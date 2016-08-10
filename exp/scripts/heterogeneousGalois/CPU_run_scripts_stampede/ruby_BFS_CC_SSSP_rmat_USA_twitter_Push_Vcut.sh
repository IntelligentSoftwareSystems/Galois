#!/bin/bash

benchmark=${2}
variant=${3}
partition=${4}
comm_mode="0"
queue=$9

algo=$benchmark"_push-"$variant"_vertex-cut"
#algo=$benchmark"_pull-"$variant"_"$partition
echo $algo

./$algo "-help"

ENV_options="MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1"

#GRAPH_rmat="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/scalefree/NEW/transpose/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.transpose.gr"
#GRAPH_usa="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/road/USA-road-d.USA-trans.gr"


GRAPH_twitter="/scratch/03279/roshand/dist-inputs/twitter-ICWSM10-component.gr"
GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/rmat28.gr" #Randomized rmat28


cmd_options_reset="-maxIterations=10000 -verify=0 -t=$1"
cmd_options="-maxIterations=10000 -verify=0 -t=$1"

if [ $benchmark = "pagerank" ]; then
	cmd_options=$cmd_options"  -tolerance=0.0000001"
fi

if [ $benchmark = "cc" ]; then
  GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/symmetric/rmat28.sgr"
  GRAPH_twitter="/scratch/03279/roshand/dist-inputs/symmetric/twitter-ICWSM10-component.sgr"
fi


if [ $5 = "rmat28" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options" -srcNodeId=155526494"
  fi

  partFileBase="/scratch/01131/rashid/inputs/partitioned"
  for i in $6
  do
    if [ $benchmark != "cc" ]; then
      partFileType="rmat28"
      if [ $i == 2 ]; then
        partFileExt="rmat28.gr"
      elif [ $i == 4 ]; then
        partFileExt="rmat28.gr"
      elif [ $i == 8 ]; then
        partFileExt="rmat28.rgr"
      elif [ $i == 16 ]; then
        partFileExt="rmat28.gr"
      elif [ $i == 32 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.rgr"
      elif [ $i == 64 ]; then
        partFileExt="rmat28.gr"
      elif [ $i == 128 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.rgr"
      elif [ $i == 256 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.rgr"
      fi
    else
      partFileExt="rmat28.sgr"
      partFileType="rmat28-sym"
    fi

    ruby ../../../../../../Distributed_latest/scripts/stampede_jobs.rb  -t "01:25:00" -q $queue -n 4 -N 4 -i dist_run_script_generated -o  ./LOG_aug_9/LOG_${algo}_TH$1\_CM${comm_mode}\_rmat28.rgr_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_rmat28  -partFolder=$partFileBase/$i/$partFileType/$partFileExt $cmd_options" -s $i  -e $i  -k 2
  done
fi



if [ $7 = "twitter" ]; then
  if [ $benchmark = "bfs" ] || [ $benchmark = "sssp" ]; then
    cmd_options=$cmd_options_reset
    cmd_options=$cmd_options" -srcNodeId=33643219"
  fi

  partFileBase="/scratch/01131/rashid/inputs/partitioned"
  for i in $8
  do
    if [ $benchmark != "cc" ]; then
       partFileExt="twitter-ICWSM10-component_withRandomWeights.gr"
       partFileType="twitter"
    else
      partFileExt="twitter-ICWSM10-component.sgr"
      partFileType="twitter-sym"
    fi

    ruby ../../../../../../Distributed_latest/scripts/stampede_jobs.rb  -t "01:25:00" -q $queue -n 4 -N 4 -i dist_run_script_generated -o  ./LOG_aug_9/LOG_${algo}_TH$1\_CM${comm_mode}\_Twitter-ICWSM10_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_twitter  -partFolder=$partFileBase/$i/$partFileType/$partFileExt $cmd_options" -s $i  -e $i  -k 2
  done
fi
