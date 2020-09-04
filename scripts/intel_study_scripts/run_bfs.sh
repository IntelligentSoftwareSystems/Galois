#!/bin/bash

echo -e "USAGE: ./run_bfs.sh <numRuns>\n"
appname=bfs

numRuns=$1
if [ -z $numRuns ]; then
  numRuns=1
fi

if [ -z ${GALOIS_BUILD} ]; then
  echo "GALOIS_BUILD not set; Please point it to the top level directory where Galois is built"
  exit
else
  echo "Using ${GALOIS_BUILD} for Galois build to run ${appname}"
fi

if [ -z ${INPUT_DIR} ]; then
  echo "INPUT_DIR not set; Please point it to the directory with .gr graphs"
  exit
else
  echo "Using ${INPUT_DIR} for inputs for ${appname}"
fi

inputDir="${INPUT_DIR}"
execDir="${GALOIS_BUILD}/lonestar/analytics/cpu/bfs"
echo ${execDir}

exec=bfs-directionopt-cpu

for configType in $(seq 1 2)
do
  if [ ${configType} == 1 ]; then
    echo "Running ${appname} with config1"
    export GOMP_CPU_AFFINITY="0-31"
    export KMP_AFFINITY="verbose,explicit,proclist=[0-31]"
    Threads=32
  else
    echo "Running ${appname} with config2"
    Threads=64
  fi

  for run in $(seq 1 ${numRuns})
  do
    for input in "kron" "road" "urand" "web" "twitter"
    do
      if [ ${input} == "web" ] || [ ${input} == "twitter" ]; then 
        ##NOTE: Using gr for directed graphs
        extension=gr
      else # kron road urand
        ##NOTE: Using sgr for undirected graphs
        extension=sgr
      fi

      if [ ${configType} == 1 ]; then 
        algo="AutoAlgo"
      elif [ ${input} == "road" ]; then # ${configType} == 2
        algo="Async"
      else # ${configType} == 2
        algo="SyncDO"
      fi

      echo "Running on ${input}"
      echo "Logs will be available in ${execDir}/logs/${input}"
      if [ ! -d "${execDir}/logs/${input}" ]; then
        mkdir -p ${execDir}/logs/${input}
      fi

      while read p; do
        source_node=$((${p} - 1))
        filename="${appname}_${input}_source_${source_node}_algo_${algo}_${configType}_Run${run}"
        statfile="${filename}.stats"
        ${execDir}/${exec} -algo=${algo} $inputDir/GAP-${input}.${extension} -t ${Threads} -preAlloc=1200  -startNode=${source_node} -statFile=${execDir}/logs/${input}/${statfile} &> ${execDir}/logs/${input}/${filename}.out
      done < $inputDir/sources/GAP-${input}_sources.mtx
    done
  done
done
