K-Core Decomposition
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Finds the <b>k-core</b> in a graph. A k-core is defined as a subgraph of a 
graph in which all vertices of degree less than k have been removed from the 
graph. The remaining vertices must all have a degree of at least k.

The algorithm supports both a bulk-synchronous and a bulk-asynchronous
parallel algorithms. This benchmark consists of two algorithms,
push- and pull-based. In the push-based algorithm, all non-removed nodes
check to see if their degree has fallen below k in each round. If so,
it removes itself and decrements the degree on its neighbors.
In the pull-based algorithm, a node will check which of its neighbors have
recently been removed from the graph and decrement its own degree in each round.
If the degree falls below k, then it removes itself from the graph.


INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs. You must specify the -symmetricGraph
flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/k-core; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 machine with a k value of 4, use the following:
`./k-core-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`
`./k-core-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`

To run on 3 hosts h1, h2, and h3 with a k value of 4, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./k-core-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`
`mpirun -n=3 -hosts=h1,h2,h3 ./k-core-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`

To run on 3 hosts h1, h2, and h3 with a k value of 4 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./k-core-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec -kcore=4`
`mpirun -n=3 -hosts=h1,h2,h3 ./k-core-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec -kcore=4`

PERFORMANCE
--------------------------------------------------------------------------------

* The push variant generally performs better in our experience.

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC) with **synchronous**
  communication for performance.

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC) with **asynchronous**
  communication for performance.
