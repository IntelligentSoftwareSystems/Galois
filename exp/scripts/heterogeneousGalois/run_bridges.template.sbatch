#!/bin/bash
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
NUM_TASKS=$3
PSET=$4
THREADS=$5

EXEC=$1
execname=$(basename "$EXEC" "")

inputname=$2
inputdirname=/pylon2/ci560jp/roshand/dist-inputs
extension=gr

FLAGS=
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdirname}/${inputname}.source`"
  fi
elif [[ ($execname == *"pagerank"*) ]]; then
  FLAGS+=" -maxIterations=125"
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=4"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
elif [[ $inputname == *"rmat"* ]]; then
  inputdirname=${inputdirname}/transpose
  extension=tgr
  FLAGS+= " -transpose"
fi
grep "${inputname}.${extension}" ${source_file}
INPUT=${inputdirname}/${inputname}.${extension}

if [ -n "$ABELIAN_VERTEX_CUT" ]; then
  FLAGS+=" -enableVertexCut" 
elif [[ ($PSET == *"gc"*) || ($PSET == *"cg"*) ]]; then
  FLAGS+=" -scalegpu=4"
fi

RUN=mpirun

source $HOME/GaloisCpp/load_modules_dist_hetero.sh

# move to working directory
WORK_DIR=/pylon1/ci4s88p/roshand/GaloisCpp
cd $WORK_DIR

set -x #echo on
MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1 $RUN -np $NUM_TASKS $EXEC ${INPUT} -pset=$PSET -t=$THREADS -num_nodes=$SLURM_NNODES $FLAGS
set +x #echo off

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $NUM_TASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

