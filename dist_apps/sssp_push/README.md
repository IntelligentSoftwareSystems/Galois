Single Source Shortest Path (Push)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program performs single source shortest path on a weighted input graph, 
starting from a source node (specified by -startNode option). 

The algorithm is a bulk-synchronous parallel version. It is push-based: in
each round, a node that has been updated from the last round will push out
its distance value to its neighbors and update them if necessary after 
considering the edge weight between itself and its neighbor.

INPUT
--------------------------------------------------------------------------------

Takes in weighted Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j sssp_push

RUN
--------------------------------------------------------------------------------

To run on 1 host with start node 0, use the following:
`./sssp_push <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 0, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./sssp_push <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 10 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./sssp_push <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -startNode=10 -partition=iec`

PERFORMANCE  
--------------------------------------------------------------------------------

Uneven load balancing among hosts can hurt performance.
