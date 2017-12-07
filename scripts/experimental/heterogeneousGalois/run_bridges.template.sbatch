#!/bin/bash
#SBATCH --mail-user=roshan@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
PART=$3
NUM_TASKS=$4
PSET=$5
THREADS=$6

execname=$1
#execdir=/pylon5/ci560jp/roshand/Galois-build/dist_apps/experimental/apps/cgo
#execdir=/pylon5/ci560jp/roshand/Galois-build/dist_apps
execdir=`pwd`
EXEC=${execdir}/${execname}

inputname=$2
inputdirname=/pylon5/ci560jp/roshand/dist-inputs
extension=gr

statname=${execname}_${inputname}_${PART}_${SLURM_NNODES}_${PSET}_${SLURM_JOB_ID}.stats

FLAGS=
FLAGS+=" -statFile=${execdir}/${statname}"
# kcore flag
if [[ $execname == *"kcore"* ]]; then
  # TODO: update this for non-100 kcore numbers
  FLAGS+=" -kcore=100"
fi
if [[ ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -startNode=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
# bc: if rmat15 is not used, specify single source flags else do
# all sources for rmat15
# TODO currently uses rmat16 (doesn't exist) so everything does single source
if [[ ($execname == *"bc"*) && ! ($inputname == "rmat16") ]]; then
  FLAGS+=" -singleSource"
  FLAGS+=" -startNode=`cat ${inputdirname}/${inputname}.source`"
fi

source_file=${inputdirname}/source
if [[ $execname == *"cc"* || $execname == *"kcore"* ]]; then
  inputdirname=${inputdirname}/symmetric
  extension=sgr
  FLAGS+=" -symmetricGraph"
else 
  # for verify purposes, always pass in graph transpose just in case it is 
  # needed for non-symmetric graphs
  FLAGS+=" -graphTranspose=${inputdirname}/transpose/${inputname}.tgr"
fi
grep "${inputname}.${extension}" ${source_file}
INPUT=${inputdirname}/${inputname}.${extension}

if [[ ($execname == *"pagerank"*) ]]; then
  FLAGS+=" -maxIterations=125"
fi
FLAGS+=" -partition=${PART}"
#if [[ ($PART == "cvc") ]]; then
#  FLAGS+=" -balanceMasters=both"
#fi
if [[ ($PART == "2dvc") ]]; then
  FLAGS+=" -balanceMasters=nodes"
fi

#if [[ ($PSET == *"cg"*) || ($PSET == *"gc"*) ]]; then
#  FLAGS+=" -scalegpu=2"
#fi

RUN=mpirun

#source $HOME/Galois/load_modules.sh

# move to working directory
WORK_DIR=/pylon5/ci560jp/roshand/Galois
cd $WORK_DIR

set -x #echo on
PSM2_MULTI_EP=1 LD_LIBRARY_PATH=$PSM2_LATEST_BUILD:$LD_LIBRARY_PATH MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1 $RUN -np $NUM_TASKS $EXEC ${INPUT} -pset=$PSET -t=$THREADS -num_nodes=$SLURM_NNODES $FLAGS
#MV2_USE_LAZY_MEM_UNREGISTER=0 MV2_ENABLE_AFFINITY=0 GALOIS_DO_NOT_BIND_THREADS=1 $RUN -np $NUM_TASKS $EXEC ${INPUT} -pset=$PSET -t=$THREADS -num_nodes=$SLURM_NNODES $FLAGS
#I_MPI_FABRICS=shm:ofa GALOIS_DO_NOT_BIND_THREADS=1 $RUN -np $NUM_TASKS $EXEC ${INPUT} -pset=$PSET -t=$THREADS -num_nodes=$SLURM_NNODES $FLAGS
set +x #echo off

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $NUM_TASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

