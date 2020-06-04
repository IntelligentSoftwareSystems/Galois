## DESCRIPTION

This benchmark computes the level of each node from a source node in an unweighted graph. It starts at a node and explores all the nodes on the same level and move on to nodes at the next depth level. 

## BUILD

Assuming CMake is performed in the ${GALOIS\_ROOT}/build, compile the application by executing the
following command in the ${GALOIS\_ROOT}/build/lonestar/analytics/gpu/bfs directory.

`make -j`

## RUN

Execute as: ./bfs [-o output-file] [-l] [-s startNode] graph-file 


The option -l  enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommneded to disable this option for high diameter graphs, such as road-networks. 

e.g., ./bfs -s 0 -o outfile.txt road-USA.gr
