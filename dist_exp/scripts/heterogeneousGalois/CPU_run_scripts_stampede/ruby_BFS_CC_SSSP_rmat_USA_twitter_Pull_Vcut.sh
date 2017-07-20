#!/bin/bash

benchmark=${2}
variant=${3}
partition=${4}
comm_mode="0"
queue=$9

algo=$benchmark"_pull-"$variant
#algo=$benchmark"_pull-"$variant"_"$partition
echo $algo

./$algo "-help"

ENV_options="MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1"

#GRAPH_rmat="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/scalefree/NEW/transpose/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.transpose.gr"
#GRAPH_usa="/work/02982/ggill0/Distributed_latest/inputs/pagerank/Galois/road/USA-road-d.USA-trans.gr"


GRAPH_twitter="/scratch/03279/roshand/dist-inputs/transpose/twitter-ICWSM10-component.tgr"

GRAPH_rmat28="/scratch/03279/roshand/dist-inputs/transpose/rmat28.tgr" #Randomized rmat28

cmd_options_reset="-maxIterations=10000 -verify=0 -t=$1 -enableVertexCut=${partition}  "
cmd_options="-maxIterations=10000 -verify=0 -t=$1 -enableVertexCut=${partition} "

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

  #partFileBase="/scratch/01131/rashid/inputs/partitioned"
  partFileBase="/scratch/02982/ggill0/dist_inputs/partitioned"
  for i in $6
  do
    if [ $benchmark != "cc" ]; then
      partFileType="rmat28-trans"
      if [ $i == 2 ]; then
        partFileExt="rmat28.tgr"
      elif [ $i == 4 ]; then
        partFileExt="rmat28.tgr"
      elif [ $i == 8 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.trgr"
      elif [ $i == 16 ]; then
        partFileExt="rmat28.tgr"
      elif [ $i == 32 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.trgr"
      elif [ $i == 64 ]; then
        partFileExt="rmat28.tgr"
      elif [ $i == 128 ]; then
        partFileExt="rmat28.tgr"
        #partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.trgr"
      elif [ $i == 256 ]; then
        partFileExt="rmat16-2e28-a=0.57-b=0.19-c=0.19-d=0.05.trgr"
      fi
    else
      partFileExt="rmat28.sgr"
      partFileType="rmat28-sym"
    fi

    ruby /work/02982/ggill0/Distributed_latest/scripts/stampede_jobs.rb  -t "01:45:00" -q $queue -n 4 -N 4 -i dist_run_script_generated -o  ./LOG_RUNS/LOG_${algo}_TH$1\_VCUT\_${partition}\_rmat28.tgr_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_rmat28  -partFolder=$partFileBase/$i/$partFileType/$partFileExt $cmd_options" -s $i  -e $i  -k 2
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
       partFileExt="twitter-ICWSM10-component_withRandomWeights.transpose.gr"
       partFileType="twitter-trans"
    else
      partFileExt="twitter-ICWSM10-component.sgr"
      partFileType="twitter-sym"
    fi

    ruby /work/02982/ggill0/Distributed_latest/scripts/stampede_jobs.rb -t "01:25:00" -q $queue -n 4 -N 4 -i dist_run_script_generated -o  ./LOG_RUNS/LOG_${algo}_TH$1\_VCUT\_${partition}\_Twitter-ICWSM10_  -A "Galois" -c "$ENV_options ibrun ./$algo $GRAPH_twitter  -partFolder=$partFileBase/$i/$partFileType/$partFileExt $cmd_options" -s $i  -e $i  -k 2
  done
fi
