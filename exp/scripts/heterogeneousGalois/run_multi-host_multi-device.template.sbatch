#!/bin/bash
#SBATCH -J het_auto_gen
#SBATCH -o het_auto_gen_%j.out
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
PSET=$3

EXEC=$1
INPUT=$2
inputdirname=$SCRATCH/inputs
#inputdirname=$(dirname "$INPUT")
inputname=$(basename "$INPUT" ".gr")
extension=gr
if [[ $EXEC == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $EXEC == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
INPUT=${inputdirname}/${inputname}.${extension}


FLAGS=""
if [[ $EXEC == *"vertex-cut"* ]]; then
  FLAGS+=" -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
else
  if [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
    FLAGS+=" -scalegpu=3"
  fi
fi

THREADS=15
if [[ $SLURM_QUEUE == "normal" ]]; then
  THREADS=16
fi

RUN=ibrun

export MV2_ENABLE_AFFINITY=0
export MV2_USE_SHARED_MEM=1
source $HOME/GaloisCpp/load_modules_dist_hetero.sh

set -x #echo on
GALOIS_DO_NOT_BIND_THREADS=1 $RUN $EXEC ${inputdirname}/${INPUT} -noverify -pset=$PSET -t=$THREADS $FLAGS
set +x #echo off

execname=$(basename "$EXEC" "")

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $SLURM_NTASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

