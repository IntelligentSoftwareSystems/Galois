EXECS=( "gcn-dist" "gcn-dist-pinned" )
#INPUTS=( "ogbn-products" )
INPUTS=( "reddit" "ogbn-products" )
#INPUTS=( "ogbn-papers100M" )
TYPES=( "sage" )
LSIZE=16
NLAYERS=2
EPOCH=200
PSET="g"

for e in "${EXECS[@]}"
do
  for t in 0
    do
    echo "Iter:"$t
    PSET="g"
    for n in 1 2 3 4
    do
      for i in "${INPUTS[@]}"
      do
        for k in "${TYPES[@]}"
        do
          TYPES_STR=${k}
          LSIZE_STR=${LSIZE}
          for nr in {1..${NLAYERS}}
          do
            TYPES_STR+=","${k}
            LSIZE_STR+=","${LSIZE}
          done
          echo GALOIS_DO_NOT_BIND_THREADS=1 mpirun -np $n ./${e} -inputDirectory=/net/ohm/export/iss/inputs/Learning/ -epochs=${EPOCH} \
                                     -layerTypes=${TYPES_STR} -disableDropout ${i} -layerSizes=${LSIZE_STR} \
                                     -numLayers=${NLAYERS} -t=56 -statFile=${e}_${i}_${k}_${LSIZE}_${NLAYERS}_${PSET}_${t}.stats -pset=${PSET} -numNodes=1


          CUDA_VISIBLE_DEVICES=2,3,4,5 GALOIS_DO_NOT_BIND_THREADS=1 mpirun -np $n ./${e} -inputDirectory=/net/ohm/export/iss/inputs/Learning/ -epochs=${EPOCH} \
                                     -layerTypes=${TYPES_STR} -disableDropout ${i} -layerSizes=${LSIZE_STR} \
                                     -numLayers=${NLAYERS} -t=56 -statFile=${e}_${i}_${k}_${LSIZE}_${NLAYERS}_${PSET}_${t}.stats -pset=${PSET} -numNodes=1
        done
      done
      PSET+="g"
      echo $PSET
    done
  done
done
