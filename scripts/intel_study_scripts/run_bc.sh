#!/bin/bash

echo -e "USAGE: ./run_bc.sh <numRuns>\n"
appname=betweennesscentrality

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
execDir="${GALOIS_BUILD}/lonestar/analytics/cpu/${appname}"
echo ${execDir}

exec=betweennesscentrality-cpu

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
        algo="Level"
      fi

      echo "Running on ${input}"
      echo "Logs will be available in ${execDir}/logs/${input}"
      if [ ! -d "${execDir}/logs/${input}" ]; then
        mkdir -p ${execDir}/logs/${input}
      fi

      for count in {0..15}
      do
        filename="${appname}_${input}_file_${count}_${configType}_Run${run}"
        statfile="${filename}.stats"
        args=" -numOfSources=4 -numOfOutSources=4 -sourcesToUse="$inputDir/sources/GAP-${input}-bc/GAP-${input}_sources_${count}.txt" "
        ${execDir}/${exec} $inputDir/GAP-${input}.${extension} -t ${Threads} ${args}  -statFile=${execDir}/logs/${input}/${statfile} &> ${execDir}/logs/${input}/${filename}.out
      done
    done
  done
done