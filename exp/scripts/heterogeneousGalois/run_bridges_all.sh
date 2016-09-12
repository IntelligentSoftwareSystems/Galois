#!/bin/sh

EXEC=$1
INPUT=$2
SET=$3
QUEUE=$4

EXECDIR=/pylon2/ci4s88p/roshand/GaloisCpp-build/exp/apps/compiler_outputs

SET="${SET%\"}"
SET="${SET#\"}"

for task in $SET; do
  IFS=",";
  set $task;
  cp run_multi-host_multi-device.template.sbatch run_multi-host_multi-device.sbatch 
  if [ $QUEUE == "GPU" ]; then
    ntasks=4
    ntasks=$((ntasks*$1))
    sed -i "2i#SBATCH -t $3" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH --gres=gpu:4" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH --ntasks-per-node 4" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -N $1" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${ntasks}_${2}_%j.out" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${ntasks}_${2}" run_multi-host_multi-device.sbatch
    threads=7
    echo -n "multi-GPU-only " $EXEC $INPUT $1 $ntasks $2 $threads $3 " "
    sbatch run_multi-host_multi-device.sbatch ${EXECDIR}/$EXEC $INPUT $ntasks $2 $threads
  elif [ $QUEUE == "GPU-shared" ]; then
    if [[ ($2 == *"gc"*) || ($2 == *"cg"*) ]]; then
      threads=28
      ngpus=$1
      ngpus=$((ngpus-1))
      sed -i "2i#SBATCH -t $3" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH --gres=gpu:$ngpus" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH --ntasks-per-node $threads" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -N 1" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_${2}_%j.out" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_${2}" run_multi-host_multi-device.sbatch
      threads=$((threads-$ngpus))
      threads=$((threads-$ngpus))
      echo -n "CPU+GPU" $EXEC $INPUT $1 $2 $threads $3 " "
      sbatch run_multi-host_multi-device.sbatch ${EXECDIR}/$EXEC $INPUT $1 $2 $threads
    else
      threads=7
      threads=$((threads*$1))
      sed -i "2i#SBATCH -t $3" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH --gres=gpu:$1" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH --ntasks-per-node $threads" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -N 1" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_${2}_%j.out" run_multi-host_multi-device.sbatch
      sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_${2}" run_multi-host_multi-device.sbatch
      echo -n "GPU-only" $EXEC $INPUT $1 $2 $threads $3 " "
      sbatch run_multi-host_multi-device.sbatch ${EXECDIR}/$EXEC $INPUT $1 $2 $threads
    fi
  elif [ $QUEUE == "RM" ]; then
    sed -i "2i#SBATCH -t $3" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH --ntasks-per-node 1" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -N $1" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_${2}_%j.out" run_multi-host_multi-device.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_${2}" run_multi-host_multi-device.sbatch
    threads=28
    echo -n "CPU-only" $EXEC $INPUT $1 $2 $threads $3 " "
    sbatch run_multi-host_multi-device.sbatch ${EXECDIR}/$EXEC $INPUT $1 $2 $threads
  fi
  rm run_multi-host_multi-device.sbatch
done

