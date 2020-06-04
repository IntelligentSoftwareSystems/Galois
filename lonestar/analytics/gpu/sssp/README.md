## DESCRIPTION

This benchmark computes the shortest path from a source node to all nodes in a directed graph with non-negative edge weights by using a modified near-far algorithm [1].

[1] https://people.csail.mit.edu/jshun/6886-s18/papers/DBGO14.pdf


## BUILD

Assuming CMake is performed in the ${GALOIS\_ROOT}/build, compile the provided the application by executing the
following command in the ${GALOIS\_ROOT}/build/lonestar/analytics/gpu/sssp directory.

`make -j`

## RUN

Execute as: ./sssp [-o output-file] [-l] [-s startNode] graph-file 


The option -l  enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommneded to disable this option for high diameter graphs, such as road-networks. 

e.g., ./sssp -s 0 -o outfile.txt road-USA.gr
