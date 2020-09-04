Single-Source Shortest Paths
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark computes the shortest path from a source node to all nodes in a directed graph with non-negative edge weights by using a modified near-far algorithm [1].

[1] https://people.csail.mit.edu/jshun/6886-s18/papers/DBGO14.pdf

INPUT
--------------------------------------------------------------------------------

Take in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/sssp; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./sssp-gpu -o <output-file> -l -s startNode <input-graph>`

-`$ ./sssp-gpu -o outfile.txt -l -s 0 rmat15.gr`

The option -l enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommended to disable this option for high diameter graphs, such as road-networks.
