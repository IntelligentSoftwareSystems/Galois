#!/bin/bash
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH -A Galois
PSET=$3

EXEC=$1
execname=$(basename "$EXEC" "")

inputname=$2
inputdirname=$SCRATCH/dist-inputs
extension=gr

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=3"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
fi
grep "${inputname}.${extension}" ${source_file}
INPUT=${inputdirname}/${inputname}.${extension}

if [ -n "$ABELIAN_VERTEX_CUT" ]; then
  FLAGS+=" -enableVertexCut -partFolder=${inputdirname}/partitions/${2}/${inputname}.${extension}"
elif [[ ($1 == *"gc"*) || ($1 == *"cg"*) ]]; then
  FLAGS+=" -scalegpu=3"
fi

THREADS=16
if [[ ($PSET == *"gc"*) || ($PSET == *"cg"*) ]]; then
  THREADS=14
fi

RUN=ibrun

source $HOME/GaloisCpp/load_modules_dist_hetero.sh

set -x #echo on
MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1 $RUN $EXEC ${INPUT} -pset=$PSET -t=$THREADS -num_nodes=$SLURM_NNODES $FLAGS
set +x #echo off

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $SLURM_NTASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

