Breadth First Search
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program performs breadth-first search on an input graph, starting from a
source node (specified by -startNode option). 

The algorithm supports both a bulk-synchronous and a bulk-asynchronous
parallel algorithms. This benchmark consists of two algorithms,
push- and pull-based. In the push-based algorithm, a node that has been
updated from the last round will push out its distance value to its neighbors
and update them if necessary in each round. In the pull-based algorithm,
every node will check its neighbors' distance values and update their own
values based on what they see in each round.

INPUT
--------------------------------------------------------------------------------

Takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/bfs; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 host with start node 0, use the following:
`./bfs-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>`
`./bfs-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>`

To run on 3 hosts h1, h2, and h3 for start node 0, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>`
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads>`

To run on 3 hosts h1, h2, and h3 for start node 10 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -startNode=10 -partition=iec`
`mpirun -n=3 -hosts=h1,h2,h3 ./bfs-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -startNode=10 -partition=iec`

PERFORMANCE
--------------------------------------------------------------------------------

* The push variant generally performs better in our experience.

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC) with **synchronous**
  communication for performance.

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC) with **asynchronous**
  communication for performance.
