#!/bin/sh

EXEC=bfs_push-topological_edge-cut

#INPUT=USA-road-d.USA.gr
INPUT=rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.gr
#INPUT=twitter-ICWSM10-component.gr
#INPUT=twitter-WWW10-component.gr
#INPUT=r4-2e25.gr

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
#QUEUE=gpu

#SET="1,1,c,3:00:00 2,2,cc,02:00:00 4,4,cccc,01:30:00"
#SET="1,1,c,2:00:00 2,2,cc,01:30:00 4,4,cccc,01:00:00"
#SET="1,1,c,1:15:00 2,2,cc,00:45:00 4,4,cccc,00:30:00"
#SET="4,4,cccc,01:00:00"
#SET="4,4,cccc,00:30:00"
#SET="2,2,cc,00:45:00 4,4,cccc,00:30:00"
#SET="1,1,c,1:15:00"
SET="1,1,c,2:00:00"
#SET="1,1,c,3:00:00"
#SET="1,1,c,4:00:00"
QUEUE=normal

EXECDIR=$WORK/GaloisCpp-build/exp/apps/het_auto_gen/

cp run_multi-host_multi-device.template.sbatch run_multi-host_multi-device.sbatch 
sed -i "7i#SBATCH -p $QUEUE" run_multi-host_multi-device.sbatch
for task in $SET; do
  IFS=",";
  set $task;
  sed -i "7i#SBATCH -N $1 -n $2" run_multi-host_multi-device.sbatch
  sed -i "7i#SBATCH -t $4" run_multi-host_multi-device.sbatch
  echo $EXEC $INPUT $1 $2 $3 $4;
  sbatch run_multi-host_multi-device.sbatch ${EXECDIR}$EXEC $INPUT $3
done
rm run_multi-host_multi-device.sbatch

