#!/bin/sh

EXEC=bfs_pull-topological_edge-cut
EXEC=sssp_pull-topological_edge-cut
EXEC=cc_pull-topological_edge-cut
EXEC=pagerank_pull-topological_edge-cut

#INPUT=road-USA
#INPUT=rmat25
#INPUT=twitter-WWW10-component
#INPUT=twitter-ICWSM10-component
INPUT=rmat28

#SET="1,1,g,4:00:00 1,2,gc,4:00:00 2,2,gg,03:00:00 2,4,gcgc,03:00:00 4,4,gggg,02:00:00 4,8,gcgcgcgc,02:00:00"
#SET="1,1,g,3:00:00 1,2,gc,3:00:00 2,2,gg,02:00:00 2,4,gcgc,02:00:00 4,4,gggg,01:30:00 4,8,gcgcgcgc,01:30:00"
#SET="1,1,g,2:00:00 1,2,gc,2:00:00 2,2,gg,01:30:00 2,4,gcgc,01:30:00 4,4,gggg,01:00:00 4,8,gcgcgcgc,01:00:00"
#SET="1,1,g,1:15:00 1,2,gc,1:15:00 2,2,gg,00:45:00 2,4,gcgc,00:45:00 4,4,gggg,00:30:00 4,8,gcgcgcgc,00:30:00"
#SET="1,2,gc,2:00:00 2,2,gg,01:30:00 2,4,gcgc,01:30:00 4,4,gggg,01:00:00"
#SET="1,2,gc,1:15:00 2,2,gg,00:45:00 2,4,gcgc,00:45:00 4,4,gggg,00:30:00"
#SET="2,2,gg,00:45:00 2,4,gcgc,00:45:00 4,4,gggg,00:30:00"
#SET="1,1,g,1:15:00"
#SET="1,1,g,3:00:00"
#SET="1,1,g,4:00:00"
SET="1,1,g,2:00:00 2,2,gg,01:30:00 4,4,gggg,01:00:00 8,8,gggggggg,00:45:00 16,16,gggggggggggggggg,00:30:00"
SET="1,1,g,2:00:00 2,2,gg,01:30:00 4,4,gggg,01:00:00 8,8,gggggggg,00:45:00 16,16,gggggggggggggggg,00:30:00 32,32,gggggggggggggggggggggggggggggggg,00:20:00"
SET="1,1,g,3:00:00 2,2,gg,02:00:00 4,4,gggg,01:30:00 8,8,gggggggg,01:00:00 16,16,gggggggggggggggg,00:45:00 32,32,gggggggggggggggggggggggggggggggg,00:30:00"
QUEUE=gpu

#SET="1,1,c,3:00:00 2,2,cc,02:00:00 4,4,cccc,01:30:00"
#SET="1,1,c,2:00:00 2,2,cc,01:30:00 4,4,cccc,01:00:00"
#SET="1,1,c,1:15:00 2,2,cc,00:45:00 4,4,cccc,00:30:00"
#SET="4,4,cccc,01:00:00"
#SET="4,4,cccc,00:30:00"
#SET="2,2,cc,00:45:00 4,4,cccc,00:30:00"
#SET="1,1,c,1:15:00"
#SET="1,1,c,2:00:00"
#SET="1,1,c,3:00:00"
#SET="1,1,c,4:00:00"
#SET="2,2,cc,0:30:00"
#SET="2,2,cc,0:30:00 4,4,cccc,0:30:00 8,8,cccccccc,0:30:00 16,16,cccccccccccccccc,0:30:00 32,32,cccccccccccccccccccccccccccccccc,0:30:00 64,64,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,0:30:00 128,128,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,0:30:00 256,256,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,0:30:00"
#SET="128,128,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,1:00:00 256,256,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,1:00:00"
#SET="128,128,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,1:00:00"
#QUEUE=normal

EXECDIR=$WORK/GaloisCpp-build/exp/apps/compiler_outputs/

for task in $SET; do
  IFS=",";
  set $task;
  cp run_multi-host_multi-device.template.sbatch run_multi-host_multi-device.sbatch 
  sed -i "2i#SBATCH -t $4" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -N $1 -n $2" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -o ${EXEC}_${INPUT}_${1}_${2}_${3}_%j.out" run_multi-host_multi-device.sbatch
  sed -i "2i#SBATCH -J $EXEC" run_multi-host_multi-device.sbatch
  echo $EXEC $INPUT $1 $2 $3 $4;
  sbatch run_multi-host_multi-device.sbatch ${EXECDIR}$EXEC $INPUT $3
  rm run_multi-host_multi-device.sbatch
done

