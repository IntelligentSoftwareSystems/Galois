PageRank (Pull)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Given a graph, ranks nodes in order of their importance using the PageRank
algorithm.

This is a bulk synchronous parallel pull based residual implementation. 
In each round, every node will pull contributions to its own pagerank from 
its neighbors nodes if they have any new contribution to give.

INPUT
--------------------------------------------------------------------------------

Takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j pagerank_pull

RUN
--------------------------------------------------------------------------------

To run on 1 host for a max of 100 iterations, use the following:
`./pagerank_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100` 

To run on 3 hosts h1, h2, and h3 for a max of 100 iterations with tolerance 0.001, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100 -tolerance=0.001`

To run on 3 hosts h1, h2, and h3 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank_pull <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -partition=iec`


PERFORMANCE  
--------------------------------------------------------------------------------

PageRank is relatively compute intensive, so load balancing computation will
matter a lot for scaling out to multiple hosts.
