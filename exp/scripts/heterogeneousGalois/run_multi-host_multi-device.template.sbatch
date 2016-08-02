#!/bin/bash
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
PSET=$3

EXEC=$1
INPUT=$2
execname=$(basename "$EXEC" "")
#inputdirname=$(dirname "$INPUT")
inputdirname=$SCRATCH/dist-inputs
#inputname=$(basename "$INPUT" ".gr")
inputname=$INPUT
extension=gr

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=10"
fi

if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
INPUT=${inputdirname}/${inputname}.${extension}

if [[ $execname == *"vertex-cut"* ]]; then
  FLAGS+=" -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
elif [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
  FLAGS+=" -scalegpu=3"
fi

THREADS=16

RUN=ibrun

export MV2_USE_LAZY_MEM_UNREGISTER=0
export MV2_ENABLE_AFFINITY=0
source $HOME/GaloisCpp/load_modules_dist_hetero.sh

set -x #echo on
GALOIS_DO_NOT_BIND_THREADS=1 $RUN $EXEC ${inputdirname}/${INPUT} -pset=$PSET -t=$THREADS $FLAGS -noverify
set +x #echo off


echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $SLURM_NTASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

