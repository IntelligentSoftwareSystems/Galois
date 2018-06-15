Breadth First Search (Pull)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program performs breadth-first search on an input graph, starting from a
source node (specified by -startNode option). 

The algorithm is a bulk-synchronous parallel version. It is pull-based: in
each round, every node will check its neighbors' distance values and update
their own values based on what they see.

INPUT
--------------------------------------------------------------------------------

Takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j bfs_pull

RUN
--------------------------------------------------------------------------------

To run on 1 host with start node 0, use the following:
`./bfs_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 0, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>` 

To run on 3 hosts h1, h2, and h3 for start node 10 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -startNode=10 -partition=iec`

PERFORMANCE
--------------------------------------------------------------------------------

The pull style version of distributed BFS generally does not perform as well as 
the push style version from our experience.

Additionally, load balancing among hosts may be an important factor to consider
when partitioning the graph.
