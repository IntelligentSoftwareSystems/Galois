#!/bin/sh

EXEC=$1
INPUT=$2
SET=$3
QUEUE=$4
PART=$5
HET=$6 # not supported for now

SET="${SET%\"}"
SET="${SET#\"}"

for task in $SET; do
  IFS=",";
  set $task;
  cp run_stampede.template.sbatch run_stampede.sbatch 
  if [ $QUEUE == "gpu" ]; then # should add HET option
    sed -i "2i#SBATCH -t $2" run_stampede.sbatch
    sed -i "2i#SBATCH -p $QUEUE" run_stampede.sbatch
    sed -i "2i#SBATCH -N $1 -n $1" run_stampede.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_g_%j.out" run_stampede.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${1}_g" run_stampede.sbatch
    threads=16
    echo "multi-GPU-only " $EXEC $INPUT $1 "g" $threads $2
    sbatch run_stampede.sbatch $EXEC $INPUT g $threads
  else
    sed -i "2i#SBATCH -t $2" run_stampede.sbatch
    sed -i "2i#SBATCH -p $QUEUE" run_stampede.sbatch
    sed -i "2i#SBATCH -N $1 -n $1" run_stampede.sbatch
    sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${PART}_${1}_c_%j.out" run_stampede.sbatch
    sed -i "2i#SBATCH -J ${EXEC}_${INPUT}_${PART}_${1}_c" run_stampede.sbatch
    threads=272
    echo "CPU-only " $EXEC $INPUT $PART $1 "c" $threads $2
    sbatch run_stampede.sbatch $EXEC $INPUT $PART c $threads 
  fi
  rm run_stampede.sbatch
done

