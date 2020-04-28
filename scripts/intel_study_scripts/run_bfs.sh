#!/bin/bash

echo -e "USAGE: ./run_pr.sh config1 2\n"
appname=bfs

if [ -z ${GALOIS_BUILD} ];
then
  echo "GALOIS_BUILD not set; Please point it to the top level directory where Galois is built"
  exit
else
  echo "Using ${GALOIS_BUILD} for Galois build to run ${appname}"
fi

if [ -z ${INPUT_DIR} ];
then
  echo "INPUT_DIR not set; Please point it to the directory with .gr graphs"
  exit
else
  echo "Using ${INPUT_DIR} for inputs for ${appname}"
fi

inputDir="${INPUT_DIR}"
execDir="${GALOIS_BUILD}/lonestar/bfs"

configType=$1
numRuns=$2

if [ -z $configType ];
then
  configType="config1"
fi
if [ -z $numRuns ];
then
  numRuns=1
fi
if [ ${configType} == "config1" ];
then
  echo "Running ${appname} with config1"
  export GOMP_CPU_AFFINITY="0-31"
  export KMP_AFFINITY="verbose,explicit,proclist=[0-31]"
  Threads=32
else
  Threads=64

fi

extension=sgr
exec=bfsDO
for run in $(seq 1 ${numRuns})
do
       for input in "kron" "road" "urand"
       do
           echo "Running on ${input}"
              if [ ${input} == "road" ];
                 then algo="Async"
                 else algo="SyncDO"
                 fi
                 echo "Logs will be available in ${execDir}/logs/${input}"
                 if [ ! -d "${execDir}/logs/${input}" ];
                  then
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

extension=gr
algo="SyncDO"
for run in $(seq 1 ${numRuns})
do
       for input in "web" "twitter"
       do
        echo "Running on ${input}"
        echo "Logs will be available in ${execDir}/logs/${input}"
        if [ ! -d "${execDir}/logs/${input}" ];
        then
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
