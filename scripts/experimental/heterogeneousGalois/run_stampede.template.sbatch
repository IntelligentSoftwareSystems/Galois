#!/bin/bash
#SBATCH --mail-user=CHANGE EMAIL HERE OR ERROR
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH -A Galois
PART=$3
PSET=$4
THREADS=$5

execname=$1
execdir=`pwd`
EXEC=${execdir}/${execname}

inputname=$2
# change this if necessary
inputdirname=/scratch/03279/roshand/dist-inputs
extension=gr

statname=${execname}_${inputname}_${PART}_${SLURM_NNODES}_${PSET}_${SLURM_JOB_ID}.stats

FLAGS=" -statFile=${execdir}/${statname}"
# kcore flag
if [[ $execname == *"kcore"* ]]; then
  FLAGS+=" -kcore=100"
fi

if [[ ($execname == *"bc"*) ||  ($execname == *"bfs"*) || ($execname == *"sssp"*) ]]; then
  if [[ -f "${inputdirname}/${inputname}.source" ]]; then
    FLAGS+=" -startNode=`cat ${inputdirname}/${inputname}.source`"
  fi
fi
if [[ ($execname == *"bc"*) ]]; then
  FLAGS+=" -singleSource"
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
  FLAGS+=" -maxIterations=100"
fi

FLAGS+=" -partition=${PART}"

################################################################################

# XTRAPULP

if [[ ($PART == "cec") ]]; then
  # idea is to balance out sends across hosts since you're sending everything
  FLAGS+=" -balanceMasters=nodes"
  if [[ $execname == *"cc"* || $execname == *"kcore"* ]]; then
    FLAGS+=" -vertexIDMapFileName=/scratch/03372/lhoang/pulp/${inputname}s_${SLURM_NNODES}.xtra"
  elif [[ $execname == *"pull"* ]]; then
    FLAGS+=" -vertexIDMapFileName=/scratch/03372/lhoang/pulp/${inputname}t_${SLURM_NNODES}.xtra"
  else # everything else, bfs, bc, sssp, push versions
    FLAGS+=" -vertexIDMapFileName=/scratch/03372/lhoang/pulp/${inputname}_${SLURM_NNODES}.xtra"
  fi
fi

################################################################################

# LOCAL GRAPH SAVES

PARTGRAPHROOT="/scratch/02681/gsg466/partitioned_graphs"

if [[ ($execname == *"bfs"*) ]]; then
  localGraphFileName="${PARTGRAPHROOT}/${PART}/${inputname}/local_graph_${inputname}_${SLURM_NNODES}"
  if [ ! -d "$localGraphFileName" ]; then
     # Control will enter here if $DIRECTORY doesn't exist.
     mkdir -p $localGraphFileName;
  fi
  FLAGS+="  -localGraphFileName=$localGraphFileName "
fi

if [[ ($execname == *"sssp"*) ]]; then
  localGraphFileName="${PARTGRAPHROOT}/${PART}/${inputname}/weighted/local_graph_${inputname}_${SLURM_NNODES}"
  if [ ! -d "$localGraphFileName" ]; then
     # Control will enter here if $DIRECTORY doesn't exist.
     mkdir -p $localGraphFileName;
  fi
  FLAGS+="  -localGraphFileName=$localGraphFileName "
  #FLAGS+=" -saveLocalGraph "
fi

if [[ $execname == *"cc"* || $execname == *"kcore"* ]]; then
  localGraphFileName="${PARTGRAPHROOT}/${PART}/${inputname}/symmetric/local_graph_${inputname}_${SLURM_NNODES}"
  if [ ! -d "$localGraphFileName" ]; then
     # Control will enter here if $DIRECTORY doesn't exist.
     mkdir -p $localGraphFileName;
  fi
  FLAGS+="  -localGraphFileName=$localGraphFileName "
  #FLAGS+=" -saveLocalGraph "
fi

if [[ ($execname == *"pagerank"*) ]]; then
  localGraphFileName="${PARTGRAPHROOT}/${PART}/${inputname}/transpose/local_graph_${inputname}_${SLURM_NNODES}"
  if [ ! -d "$localGraphFileName" ]; then
     # Control will enter here if $DIRECTORY doesn't exist.
     mkdir -p $localGraphFileName;
  fi
  FLAGS+="  -localGraphFileName=$localGraphFileName "
  #FLAGS+=" -saveLocalGraph "
fi

if [[ ($execname == *"partition"*) ]]; then
  localGraphFileName="${PARTGRAPHROOT}/${PART}/${inputname}/local_graph_${inputname}_${SLURM_NNODES}"
  if [ ! -d "$localGraphFileName" ]; then
     # Control will enter here if $DIRECTORY doesn't exist.
     mkdir -p $localGraphFileName;
  fi
  FLAGS+="  -localGraphFileName=$localGraphFileName "
  #FLAGS+=" -saveLocalGraph "
fi

if [[ -n "$SAVE_GRAPH" ]]; then
  FLAGS+=" -saveLocalGraph "
fi

if [[ -n "$LOAD_GRAPH" ]]; then
  FLAGS+=" -readFromFile "
fi

RUN=ibrun

set -x #echo on
PRINT_PER_HOST_STATS=1 GALOIS_DO_NOT_BIND_THREADS=1 $RUN $EXEC ${INPUT} -t=$THREADS $FLAGS -edgeBufferSize=8388608 -runs=3
set +x #echo off

# give permissions to output files
chmod 660 ${execname}_${inputname}_${PART}_${SLURM_NNODES}_${PSET}_${SLURM_JOB_ID}.out
chmod 660 $statname

echo "Algorithm: " $execname
echo "Input: " $INPUT
echo "Number of nodes: " $SLURM_NNODES
echo "Number of tasks: " $SLURM_NTASKS
echo "Number of tasks per node: " $SLURM_TASKS_PER_NODE
echo "Devices: " $PSET
