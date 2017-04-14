#!/bin/bash
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
NUM_TASKS=$3
PSET=$4
THREADS=$5

execname=$1
execdir=/pylon2/ci4s88p/roshand/GaloisCpp-build-lwci/exp/apps/compiler_outputs
EXEC=${execdir}/${execname}

inputname=$2
inputdir=/pylon2/ci560jp/roshand/dist-inputs
extension=gr

statname=${execname}_${inputname}_${SLURM_NNODES}_${PSET}_${SLURM_JOB_ID}.stats

FLAGS=" -statOutputFile=${execdir}/${statname}"
FLAGS=
#FLAGS=" -runs=1"
#FLAGS+=" -numPipelinedPhases=4"
#FLAGS+=" -numPipelinedPhases=${NUM_TASKS}"
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdir}/${inputname}.source" ]]; then
    FLAGS+=" -srcNodeId=`cat ${inputdir}/${inputname}.source`"
  fi
fi

source_file=${inputdir}/source
if [[ $execname == *"cc"* ]]; then
  inputdir=${inputdir}/symmetric
  extension=sgr
elif [[ $execname == *"pull"* ]]; then
  inputdir=${inputdir}/transpose
  extension=tgr
elif [[ $inputname == *"rmat"* ]]; then
#else
  inputdir=${inputdir}/transpose
  extension=tgr
  FLAGS+=" -transpose"
fi
grep "${inputname}.${extension}" ${source_file}
INPUT=${inputdir}/${inputname}.${extension}

if [ -n "$ABELIAN_VERTEX_CUT" ]; then
  FLAGS+=" -enableVertexCut" 
elif [[ ($PSET == *"gc"*) || ($PSET == *"cg"*) ]]; then
  FLAGS+=" -scalegpu=4"
fi
if [[ $execname == *"worklist"* ]]; then
  FLAGS+=" -cuda_wl_dup_factor=4"
fi
if [[ ($execname == *"pagerank"*) ]]; then
  FLAGS+=" -maxIterations=125"
fi

RUN=mpirun

source $HOME/GaloisCpp/load_modules_dist_hetero.sh

# move to working directory
WORK_DIR=/pylon5/ci4s88p/roshand/GaloisCpp
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

