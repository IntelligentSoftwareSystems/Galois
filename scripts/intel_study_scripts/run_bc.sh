#!/bin/bash

echo -e "USAGE: ./run_pr.sh config1 2\n"
appname=betweennesscentrality

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
  echo "Using directory ${INPUT_DIR} for inputs for ${appname}"
fi

inputDir="${INPUT_DIR}"
execDir="${GALOIS_BUILD}/lonestar/${appname}"

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
for run in $(seq 1 ${numRuns})
do
       for input in "kron" "road" "urand"
       do
           echo "Running on ${input}"
           if [ ${input} == "road" ];
           then
             exec="bc-async"
           else exec="bc-level"
           fi
           echo "Logs will be available in ${execDir}/logs/${input}"
           if [ ! -d "${execDir}/logs/${input}" ];
            then
              mkdir -p ${execDir}/logs/${input}
           fi
           while read p; do
              source_node=$((${p} - 1))
              filename="${appname}_${input}_source_${source_node}_${configType}_Run${run}"
              statfile="${filename}.stats"
              if [ ${input} == "road" ];
              then
                args=" -numOfSources=1 -numOfOutSources=1 -sourcesToUse="${source_node}" "
              else
                args=" -numOfSources=1 -singleSource  -startNode=${source_node}"
              fi
              ${execDir}/${exec} $inputDir/GAP-${input}.${extension} -t ${Threads} ${args}  -statFile=${execDir}/logs/${input}/${statfile} &> ${execDir}/logs/${input}/${filename}.out
               done < $inputDir/sources/GAP-${input}_sources.mtx
       done
done

extension=gr
exec="bc-level"
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
          filename="${appname}_${input}_source_${source_node}_${configType}_Run${run}"
          statfile="${filename}.stats"
          args=" -numOfSources=1 -singleSource  -startNode=${source_node}"
          ${execDir}/${exec} $inputDir/GAP-${input}.${extension} -t ${Threads} ${args}  -statFile=${execDir}/logs/${input}/${statfile} &> ${execDir}/logs/${input}/${filename}.out
          done < $inputDir/sources/GAP-${input}_sources.mtx
        done
done
