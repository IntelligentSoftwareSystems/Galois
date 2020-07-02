#!/bin/bash

echo -e "USAGE: ./run_pr.sh <numRuns>\n"
appname="pagerank"

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
if [ ! -d "${execDir}/logs/" ]; then
  mkdir -p ${execDir}/logs/
fi
echo "Logs will be available in ${execDir}/logs/"

exec="pagerank-pull-cpu"

algo="Topo"
tol=1e-4
maxIter=1000

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
        ##NOTE: Using tgr for directed graphs
        extension=tgr
      else # kron road urand
        ##NOTE: Using sgr for undirected graphs
        extension=sgr
      fi
      echo "Running on ${input}"
      filename="${appname}_${input}_algo_${algo}_${configType}_Run${run}"
      statfile="${filename}.stats"
      ${execDir}/${exec} -algo=$algo -t=${Threads} $inputDir/GAP-${input}.${extension} -tolerance=${tol} -maxIterations=${maxIter} -transposedGraph -statFile=${execDir}/logs/${statfile} &> ${execDir}/logs/${filename}.out
    done
  done
done
