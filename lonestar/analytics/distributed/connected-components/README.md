Connected Components
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Find all connected components of an undirected (symmetric) graph. Set the same 
label to nodes which belong to the same component.

The algorithm supports both a bulk-synchronous and a bulk-asynchronous
parallel algorithms. This benchmark consists of two algorithms,
push- and pull-based.  In the push variant of the algorithm, nodes with a label
that has changed from the last round will push this label out to its neighbors
and update their labels with a min operation. In the pull variant of the
algorithm, all nodes check their neighbors to see if they have a lower label,
and they will adopt the lowest label among its neighbors/itself as its component.

INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs. You must specify the -symmetricGraph
flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/connected-components/; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 machine, use the following:
`./connected-components-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph`
`./connected-components-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./connected-components-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph`
`mpirun -n=3 -hosts=h1,h2,h3 ./connected-components-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./connected-components-push-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec`
`mpirun -n=3 -hosts=h1,h2,h3 ./connected-components-pull-dist <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec`

PERFORMANCE
--------------------------------------------------------------------------------

* The push variant generally performs better in our experience.

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC) with **synchronous**
  communication for performance.

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC) with **asynchronous**
  communication for performance.
