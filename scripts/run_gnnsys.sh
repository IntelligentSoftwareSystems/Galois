#!/bin/bash

INPUTDIR="/net/ohm/export/iss/inputs/Learning/"
#EXECS=( "gcn-dist" "gcn-dist-pinned" )
EXECS=( "gcn-dist" )
#INPUTS=( "cora" "reddit" "ogbn-products" )
INPUTS=( "reddit" ) 
LAYERTYPES=( "sage" "gcn" )
#LAYERTYPES=( "gcn" )
LAYERSIZE=16
NUMLAYERS=2
#PARTITIONS=( "oec" "cvc" )
PARTITIONS=( "oec" )
DEVICES="0"

FLAGS=" -epochs=200"
#FLAGS+=" -disableDropout"
#FLAGS+=" -testInterval=50"

PREFIX="GALOIS_DO_NOT_BIND_THREADS=1 "

for input in "${INPUTS[@]}"
do
  for partition in "${PARTITIONS[@]}"
  do
#for num_gpus in {2..4}
    for num_gpus in 1
    do
      PSET="g"
      for ngpus in $(seq 2 ${num_gpus})
      do
        PSET+="g"
      done
      for layer in "${LAYERTYPES[@]}"
      do
        for exe in "${EXECS[@]}"
        do
          # Variable parameters
          LSIZE_STR=$LAYERSIZE
          LTYPE_STR=$layer
          for r in {1..${NUMLAYERS}} 
          do
            LSIZE_STR+=","$LAYERSIZE
            LTYPE_STR+=","$layer
          done
          echo "CUDA_VISIBLE_DEVICES=${DEVICES} GALOIS_DO_NOT_BIND_THREADS=1 mpirun -np $num_gpus ./${exe} $input $FLAGS -layerTypes=${LTYPE_STR} -t=1 \
                            -pset=${PSET} -layerSizes=${LSIZE_STR} -numNodes=1 --inputDirectory=${INPUTDIR} \
                            -statFile=${exe}_${input}_${layer}_${NUMLAYERS}_${LAYERSIZE}_${PSET}_${partition}.stat -partition=${partition}"

          CUDA_VISIBLE_DEVICES=${DEVICES} GALOIS_DO_NOT_BIND_THREADS=1 mpirun -np $num_gpus ./${exe} $input $FLAGS -layerTypes=${LTYPE_STR} -t=1 \
                                    -pset=${PSET} -layerSizes=${LSIZE_STR} -numNodes=1 --inputDirectory=${INPUTDIR} \
                                    -statFile=${exe}_${input}_${layer}_${NUMLAYERS}_${LAYERSIZE}_${PSET}_${partition}.stat -partition=${partition}
        done
      done
    done
  done
done
