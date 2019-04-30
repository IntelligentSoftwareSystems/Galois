K-Core Decomposition (Push)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Finds the <b>k-core</b> in a graph. A k-core is defined as a subgraph of a 
graph in which all vertices of degree less than k have been removed from the 
graph. The remaining nodes must all have a degree of at least k.

This is a bulk synchronous parallel push-style implementation. In each round,
all non-removed nodes check to see if their degree has fallen below k. If so,
it removes itself and decrements the degree on its neighbors.

INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs. You must specify the -symmetricGraph
flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j kcore_push

RUN
--------------------------------------------------------------------------------

To run on 1 machine with a k value of 4, use the following:
`./kcore_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`

To run on 3 hosts h1, h2, and h3 with a k value of 4, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./kcore_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph -kcore=4`

To run on 3 hosts h1, h2, and h3 with a k value of 4 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./kcore_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec -kcore=4`

PERFORMANCE
--------------------------------------------------------------------------------

Uneven load balancing among hosts can heavily affect performance.
