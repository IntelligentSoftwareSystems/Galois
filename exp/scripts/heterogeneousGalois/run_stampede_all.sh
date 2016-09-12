#!/bin/sh

EXEC=$1
INPUT=$2
SET=$3
QUEUE=gpu

EXECDIR=$WORK/GaloisCpp-build/exp/apps/compiler_outputs/

SET="${SET%\"}"
SET="${SET#\"}"

for task in $SET; do
  IFS=",";
  set $task;
  cp run_multi-host_multi-device.template.sbatch run_multi-host_multi-device.sbatch 
  sed -i "2i#SBATCH -t $4" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -N $1 -n $2" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_${2}_${3}_%j.out" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -J ${EXEC}_${1}_${2}" run_multi-host_multi-device.sbatch
  echo $EXEC $INPUT $1 $2 $3 $4;
  sbatch run_multi-host_multi-device.sbatch ${EXECDIR}$EXEC $INPUT $3
  rm run_multi-host_multi-device.sbatch
done

