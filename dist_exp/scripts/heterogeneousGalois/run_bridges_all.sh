#!/bin/sh

EXEC=$1
INPUT=$2
SET=$3
QUEUE=$4
PART=$5
#HET=$6 # not supported for now

current_dir=$(dirname "$0")

SET="${SET%\"}"
SET="${SET#\"}"

for task in $SET; do
  IFS=",";
  set $task;
  cp $current_dir/run_bridges.template.sbatch $current_dir/run_bridges.sbatch 
  if [ $QUEUE == "GPU" ]; then # should add HET option
    ntasks=4
    ntasks=$((ntasks*$1))
    sed -i "2i#SBATCH -t $2" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH --gres=gpu:k80:4" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH --ntasks-per-node 4" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -N $1" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -p $QUEUE" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_gggg_%j.out" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_gggg" $current_dir/run_bridges.sbatch
    threads=7
    echo -n "multi-GPU-only " $EXEC $INPUT $1 $ntasks "gggg" $threads $2 " "
    sbatch $current_dir/run_bridges.sbatch $EXEC $INPUT $ntasks gggg $threads
  elif [ $QUEUE == "GPU-shared" ]; then # should be fixed
    #if [[ $HET == 1 ]]; then
      threads=28
      ngpus=$1
      ngpus=$((ngpus-1))
      sed -i "2i#SBATCH -t $2" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH --gres=gpu:$ngpus" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH --ntasks-per-node $threads" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH -N 1" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH -p $QUEUE" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_cg_%j.out" $current_dir/run_bridges.sbatch
      sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_cg" $current_dir/run_bridges.sbatch
      threads=$((threads-$ngpus))
      threads=$((threads-$ngpus))
      echo -n "CPU+GPU" $EXEC $INPUT $1 "cgggg" $threads $2 " "
      sbatch $current_dir/run_bridges.sbatch $EXEC $INPUT $1 cgggg $threads
    #else
    #  threads=7
    #  threads=$((threads*$1))
    #  sed -i "2i#SBATCH -t $2" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH --gres=gpu:$1" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH --ntasks-per-node $threads" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH -N 1" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH -p $QUEUE" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_g_%j.out" $current_dir/run_bridges.sbatch
    #  sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_g" $current_dir/run_bridges.sbatch
    #  echo -n "GPU-only" $EXEC $INPUT $1 "gggg" $threads $2 " "
    #  sbatch $current_dir/run_bridges.sbatch $EXEC $INPUT $1 gggg $threads
    #fi
  elif [ $QUEUE == "RM" ]; then
    sed -i "2i#SBATCH -t $2" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH --ntasks-per-node 1" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -N $1" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -p $QUEUE" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${PART}_${1}_c_%j.out" $current_dir/run_bridges.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${PART}_${1}_c" $current_dir/run_bridges.sbatch
    threads=28
    echo -n "CPU-only" $EXEC $INPUT $PART $1 "c" $threads $2 " "
    sbatch $current_dir/run_bridges.sbatch $EXEC $INPUT $PART $1 c $threads
  fi
  rm $current_dir/run_bridges.sbatch
done

