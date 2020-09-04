Breadth-First Search
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark computes the level of each node from a source node in an unweighted graph. It starts at a node and explores all the nodes on the same level and move on to nodes at the next depth level.

INPUT
--------------------------------------------------------------------------------

Take in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/bfs; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./bfs-gpu -o <output-file> -l -s <startNode> <input-graph>`

-`$ ./bfs-gpu -o outfile.txt -l -s 0 rmat15.gr`

The option -l enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommended to disable this option for high diameter graphs, such as road-networks.
