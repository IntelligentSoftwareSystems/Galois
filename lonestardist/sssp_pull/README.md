Single Source Shortest Path (Pull)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program performs single source shortest path on a weighted input graph, 
starting from a source node (specified by -startNode option). 

The algorithm is a bulk-synchronous parallel version. It is pull-based: in
each round, every node will check its neighbors' distance values and update
their own values based on the edge weight between the node and its neighbor.

INPUT
--------------------------------------------------------------------------------

Takes in weighted Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j sssp_pull

RUN
--------------------------------------------------------------------------------

To run on 1 host with start node 0, use the following:
`./sssp_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 0, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./sssp_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 10 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./sssp_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -startNode=10 -partition=iec`

PERFORMANCE  
--------------------------------------------------------------------------------

In our experience, the push version generally performs beter than the pull version.

Uneven load balancing among hosts can hurt performance.
