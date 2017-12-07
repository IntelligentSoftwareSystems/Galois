#!/bin/bash
#SBATCH --mail-user=gill@cs.utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH -A Galois
#TG-CIE170005
PART=$3
PSET=$4
THREADS=$5

execname=$1
#execdir=$WORK/Galois-build/dist_apps
execdir=/scratch/02681/gsg466/build_Galois_master/release_lwci/dist_apps
EXEC=${execdir}/${execname}

inputname=$2
inputdirname=/scratch/03279/roshand/dist-inputs #$SCRATCH/dist-inputs
extension=gr

statname=${execname}_${inputname}_${PART}_${SLURM_NNODES}_${PSET}_${SLURM_JOB_ID}.stats

FLAGS=" -statFile=${execdir}/${statname}"
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
grep "${inputname}.${extension}" ${source_file} >>$LOG
INPUT=${inputdirname}/${inputname}.${extension}

if [[ ($execname == *"pagerank"*) ]]; then
  FLAGS+=" -maxIterations=100"
fi
FLAGS+=" -partition=${PART}"
#if [[ ($PART == "cvc") ]]; then
#  FLAGS+=" -balanceMasters=both"
#fi
if [[ ($PART == "2dvc") ]]; then
  FLAGS+=" -balanceMasters=nodes"
fi
if [[ ($PART == "cec") ]]; then
  FLAGS+=" -balanceMasters=nodes"
  FLAGS+=" -vertexIDMapFileName=/scratch/02681/gsg466/xtraPulp_output/rmat28.gr_16.output"
fi

RUN=ibrun

#source $HOME/galois/load_modules.sh

# move to working directory
WORK_DIR=$SCRATCH/Galois
cd $WORK_DIR

set -x #echo on
PRINT_PER_HOST_STATS=1 GALOIS_DO_NOT_BIND_THREADS=1 $RUN $EXEC ${INPUT} -t=$THREADS $FLAGS
set +x #echo off

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $SLURM_NTASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET

