PageRank
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Given a graph, ranks nodes in order of their importance using the PageRank
algorithm.

The algorithm supports both a bulk-synchronous and a bulk-asynchronous
parallel algorithms. This benchmark consists of two algorithms,
push- and pull-based. In the push-based algorithm, if a node has new
contributions to its neighbors' page rank values, it will push them out
to them, in each round. In the pull-based algorithm, every node will
contribute to its own pagerank from its neighbors if they have any new
contributions to give, in each round.

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/pagerank/; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 host for a max of 100 iterations, use the following:
`./pagerank-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100` 
`./pagerank-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100` 

To run on 3 hosts h1, h2, and h3 for a max of 100 iterations with tolerance 0.001, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100 -tolerance=0.001`
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -maxIterations=100 -tolerance=0.001`

To run on 3 hosts h1, h2, and h3 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank-push-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -partition=iec`
`mpirun -n=3 -hosts=h1,h2,h3 ./pagerank-pull-dist <input-graph> -graphTranspose=<transpose-input-graph> -t=<num-threads> -partition=iec`

PERFORMANCE
--------------------------------------------------------------------------------

* The pull variant generally performs better in our experience.

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC) with **synchronous**
  communication for performance.

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC) with **asynchronous**
  communication for performance.
